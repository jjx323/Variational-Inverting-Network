#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:14:04 2021

@author: jjx323
"""

from glob import glob
import warnings
import time
import random
import numpy as np
import shutil
import torchvision.utils as vutils
from utils import batch_PSNR, batch_SSIM
from tensorboardX import SummaryWriter
from math import ceil
from loss import loss_fn_residual, loss_fn_clean
from networks import VDN
from datasets import InvertingDatasets
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
from pathlib import Path
from options import set_opts
import fenics as fe
from scipy import sparse
import h5py as h5
import cv2
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
from core.model import Domain2D
from core.misc import trans_to_python_sparse_matrix
from SimpleSmooth.common import EquSolver
from ModelN import gene_vec2matrix, gene_matrix2vec 


# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

args = set_opts()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

_lr_min = 1e-6
_modes = ['train', 'test']

# _lr_min = 1e-8
# args.lr = 1e-6
args.batch_size = 32
args.learn_type = 'clean'
args.chn = 1

## ----------------------------------------------------------------------------
## set the forward operator

class ForwardOP(object):
    def __init__(self, mesh_size, smooth_para=0.1, resize=False, \
                 coor_path='./data/coordinates.npy'):
        self.mesh_size = mesh_size
        self.mesh = fe.UnitSquareMesh(mesh_size-1, mesh_size-1)
        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        self.ll = mesh_size*mesh_size
        self.Id_matrix = sparse.coo_matrix(sparse.eye(self.ll))    
        equ_nx, equ_ny = mesh_size-1, mesh_size-1
        self.domain_equ = Domain2D(nx=equ_nx, ny=equ_ny, mesh_type='P', mesh_order=1)
        coef_nx, coef_ny = mesh_size-1, mesh_size-1
        self.domain_coef = Domain2D(nx=coef_nx, ny=coef_ny, mesh_type='P', mesh_order=1)
        self.smooth_para = smooth_para
        self.coordinates = np.load(coor_path)
        fun = fe.Function(self.domain_coef.function_space)
        self.equ_solver = EquSolver(domain_equ=self.domain_equ, domain_coef=self.domain_coef, \
                                    alpha=self.smooth_para, points=np.array(self.coordinates), \
                                    m=fun)
        self.M = sps.csr_matrix(trans_to_python_sparse_matrix(self.equ_solver.M_))
        self.forward_matrix = sps.csr_matrix(trans_to_python_sparse_matrix(self.equ_solver.Finv_))
        self.S_forward = self.equ_solver.S
        self.len_data, _ = self.S_forward.shape
        self.l_data = np.int(np.sqrt(self.len_data))
        self.resize = resize
        self.matrix2vec = gene_matrix2vec(self.domain_coef.function_space)
        self.vec2matrix = gene_vec2matrix(self.domain_coef.function_space)
    
    def forward_op(self, z):
        batch_size, ch, _, _ = z.shape
        if z.is_cuda:
            z_np = self.matrix2vec(z.detach().cpu().numpy())
        else:
            z_np = self.matrix2vec(z.detach().numpy())
        z_np = z_np.reshape(-1, self.ll).transpose((1, 0))
        out = (self.S_forward@spsl.spsolve(self.forward_matrix, self.M@z_np)).reshape(self.len_data, -1)
        # out = (spsl.spsolve(self.forward_matrix, self.M@z_np)).reshape(self.ll, -1)
        out = out.transpose((1, 0))
        out = out.reshape(-1, args.chn, self.l_data, self.l_data)
        # out = self.vec2matrix(out.reshape(batch_size, ch, self.ll))
        if self.resize == True:
            channel, chn, _, _ = out.shape
            out_final = np.zeros((channel, chn, self.mesh_size, self.mesh_size))
            for ii in range(channel):
                for jj in range(chn):
                    out_final[ii, jj, :] = cv2.resize(out[ii, jj, :], \
                                             (self.mesh_size, self.mesh_size), \
                                             interpolation=cv2.INTER_AREA)
        else:
            out_final = out
            
        out_final = torch.from_numpy(out_final).to(device=z.device).type(dtype=z.dtype)
        return out_final

smooth_para = 0.1   
forwardOP_full = ForwardOP(args.patch_size, smooth_para=smooth_para, resize=True)
print("Preparations of the forward operator are completed!")

## set the forward operator
## ----------------------------------------------------------------------------

## ----------------------------------------------------------------------------
## train_model

def train_model(net, datasets, optimizer, lr_scheduler, criterion, \
                forward_op=None, epoch_num=args.epochs, log_dir=args.log_dir,  \
                model_dir=args.model_dir):
    C = args.chn
    clip_grad_D, clip_grad_I, clip_grad_S = args.clip_grad_D, args.clip_grad_I, args.clip_grad_S
    data_loader = {phase: torch.utils.data.DataLoader(datasets[phase], \
                                batch_size=args.batch_size, shuffle=True,\
                                num_workers=args.num_workers, pin_memory=True) \
                                for phase in datasets.keys()}
    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase]/args.batch_size) for phase in datasets.keys()}
    writer = SummaryWriter(log_dir)
    if args.resume:
        step = args.step
        step_fun = args.step_fun
    else:
        step = 0
        step_fun = {x: 0 for x in _modes}
    param_D = [x for name, x in net.named_parameters() if 'dnet' in name.lower()]
    param_S = [x for name, x in net.named_parameters() if 'snet' in name.lower()]
    
    for epoch in range(args.epoch_start, epoch_num):
        loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLG', 'KLIG']}
        mse_per_epoch = {x: 0 for x in _modes}
        grad_norm_D = grad_norm_S = 0
        tic = time.time()
        # train stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        if lr < _lr_min:
            sys.exit('Reach the minimal learning rate')
        phase = 'train'
        
        for ii, data in enumerate(data_loader[phase]):
            if type(matrix_HTH) == type(None):
                d_noise, d_clean, u, sigmaMapEst, sigmaMapGt = [x.cuda() for x in data]
            else:
                ## need further updatate
                d_noise_HTH, u, sigmaMapEst, sigmaMapGt = [x.cuda() for x in data]
                d_noise = d_noise_HTH[:, :args.chn, :, :]
                
            optimizer.zero_grad()
            if type(matrix_HTH) == type(None):
                phi_Z, phi_sigma = net(d_noise, 'train')
            else:
                ## needs further update
                phi_Z, phi_sigma = net(d_noise_HTH, 'train')
            
            _, _, lu, ru = u.shape
            d_noise = F.interpolate(d_noise, size=[lu, ru], mode='bicubic')
            sigmaMapEst = F.interpolate(sigmaMapEst, size=[lu, ru], mode='bicubic')
            sigmaMapGt = F.interpolate(sigmaMapGt, size=[lu, ru], mode='bicubic')
            loss, g_lh, kl_g, kl_Igam = criterion(phi_Z, phi_sigma, d_noise, u, sigmaMapGt, \
                                                      args.eps2, radius=args.radius,\
                                                      forward_op=forward_op)
            
            loss.backward()
            # clip the gradnorm
            total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)
            total_norm_S = nn.utils.clip_grad_norm_(param_S, clip_grad_S)
            grad_norm_D = (grad_norm_D*(ii/(ii+1)) + total_norm_D/(ii+1))
            grad_norm_S = (grad_norm_S*(ii/(ii+1)) + total_norm_S/(ii+1))
            optimizer.step()
        
            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            loss_per_epoch['lh'] += g_lh.item() / num_iter_epoch[phase]
            loss_per_epoch['KLG'] += kl_g.item() / num_iter_epoch[phase]
            loss_per_epoch['KLIG'] += kl_Igam.item() / num_iter_epoch[phase]
            
            u_invert = phi_Z[:, :C,].detach().data
            mse = F.mse_loss(u_invert, u)
            mse_per_epoch[phase] += mse
            if (ii+1)%args.print_freq == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, lh={:+4.2f}, ' + \
                                'KLG={:+>7.2f}, KLIG={:+>6.2f}, mse={:.2e}, ' + \
                                    'GNorm_D:{:.1e}/{:.1e}, ' + \
                                   'GNorm_S:{:.1e}/{:.1e}, lr={:.1e}'
                print(log_str.format(epoch+1, args.epochs, phase, ii+1, num_iter_epoch[phase], \
                                         g_lh.item(), kl_g.item(), kl_Igam.item(), mse, \
                                         clip_grad_D, total_norm_D, \
                                         clip_grad_S, total_norm_S, lr))
                writer.add_scalar('Train Loss Iter', loss.item(), step)
                writer.add_scalar('Train MSE Iter', mse, step)
                writer.add_scalar('Gradient Norm_D Iter', total_norm_D, step)
                writer.add_scalar('Gradient Norm_S Iter', total_norm_S, step)
                step += 1
            if (ii+1) % (5*args.print_freq) == 0:
                alpha = torch.exp(phi_sigma[:, :C, ])
                beta = torch.exp(phi_sigma[:, C:, ])
                sigmaMap_pred = beta / (alpha-1)
                x1 = vutils.make_grid(u_invert, normalize=True, scale_each=True)
                writer.add_image(phase+' Estimated function u', x1, step_fun[phase])
                x2 = vutils.make_grid(u, normalize=True, scale_each=True)
                writer.add_image(phase+' GroundTruth', x2, step_fun[phase])
                x3 = vutils.make_grid(sigmaMap_pred, normalize=True, scale_each=True)
                writer.add_image(phase+' Predict Sigma', x3, step_fun[phase])
                x4 = vutils.make_grid(sigmaMapGt, normalize=True, scale_each=True)
                writer.add_image(phase+' Groundtruth Sigma', x4, step_fun[phase])
                x5 = vutils.make_grid(d_noise, normalize=True, scale_each=True)
                writer.add_image(phase+' Noisy Data', x5, step_fun[phase])
                step_fun[phase] += 1
        
        mse_per_epoch[phase] /= (ii+1)
        log_str ='{:s}: Loss={:+.2e}, lh={:+.2e}, KL_Guass={:+.2e}, KLIG={:+.2e}, mse={:.3e}, ' + \
                  'GNorm_D={:.1e}/{:.1e}, GNorm_S={:.1e}/{:.1e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['lh'],
                                loss_per_epoch['KLG'], loss_per_epoch['KLIG'], mse_per_epoch[phase],
                                clip_grad_D, grad_norm_D, \
                                clip_grad_S, grad_norm_S))
        writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
        writer.add_scalar('Mean Grad Norm_D epoch', grad_norm_D, epoch)
        writer.add_scalar('Mean Grad Norm_S epoch', grad_norm_S, epoch)
        clip_grad_D = min(clip_grad_D, grad_norm_D)
        clip_grad_S = min(clip_grad_S, grad_norm_S)
        print('-'*150)
        
        # test stage
        net.eval()
        phase = 'test'
        # d_clean = np.load('./data/u_128.npy')  # just for testing
        d_clean = np.load('./data/d_clean.npy')
        d_len = np.int(np.sqrt(len(d_clean)))
        # d_len, _ = d_clean.shape
        d_clean = d_clean.reshape(d_len, d_len)
        u = np.load('./data/u_128.npy')
        lu, ru = u.shape
        
        noise_level = 0.01*np.max(np.abs(d_clean))
        d_noise = d_clean + noise_level*np.random.randn(d_len, d_len)
        d_noise = cv2.resize(d_noise, (lu, ru), interpolation=cv2.INTER_AREA)      
        u = torch.tensor(u[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
           
        if type(matrix_HTH) == type(None):
            d_noise = torch.tensor(d_noise[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
            d_clean = torch.tensor(d_clean[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
            d_noise, d_clean, u = d_noise.cuda(), d_clean.cuda(), u.cuda()
            with torch.set_grad_enabled(False):
                phi_Z, phi_sigma = net(d_noise, 'train')
        else:
            # needs further update
            d_noise_HTH = np.concatenate((d_noise, matrix_HTH), axis=0)
            d_noise_HTH = torch.tensor(d_noise_HTH[np.newaxis, :, :, :], dtype=torch.float32)
            d_noise_HTH, u = d_noise_HTH.cuda(), u.cuda()
            with torch.set_grad_enabled(False):
                phi_Z, phi_sigma = net(d_noise_HTH, 'train')
        
        u_invert = phi_Z[:, :C,].detach().data
        mse = F.mse_loss(u_invert, u)
        print("Test: mse = {:.2e}".format(mse))
        
        # tensorboardX summary
        alpha = torch.exp(phi_sigma[:, :C, ])
        beta = torch.exp(phi_sigma[:, C:, ])
        sigmaMap_pred = beta / (alpha-1)
        x1 = vutils.make_grid(u_invert, normalize=True, scale_each=True)
        writer.add_image(phase+' Inverted u', x1, step_fun[phase])
        x2 = vutils.make_grid(u, normalize=True, scale_each=True)
        writer.add_image(phase+' GroundTruth', x2, step_fun[phase])
        x3 = vutils.make_grid(sigmaMap_pred, normalize=True, scale_each=True)
        writer.add_image(phase+' Predict Sigma', x3, step_fun[phase])
        x4 = vutils.make_grid(d_noise, normalize=True, scale_each=True)
        writer.add_image(phase+' Noise data', x4, step_fun[phase])
        # x5 = vutils.make_grid(sigmaMap_gt, normalize=True, scale_each=True)
        # writer.add_image(phase+' True Sigma', x5, step_fun[phase])
        step_fun[phase] += 1
    
        lr_scheduler.step()
        # save model
        if (epoch+1) % args.save_model_freq == 0 or epoch+1 == args.epochs:
            model_prefix = 'model_'
            save_path_model = os.path.join(model_dir, model_prefix+str(epoch+1))
            torch.save({
                'epoch': epoch+1,
                'step': step+1,
                'step_img': {x: step_fun[x] for x in _modes},
                'grad_norm_D': clip_grad_D,
                'grad_norm_S': clip_grad_S,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, save_path_model)
            model_state_prefix = 'model_state_'
            save_path_model_state = os.path.join(model_dir, model_state_prefix+str(epoch+1))
            torch.save(net.state_dict(), save_path_model_state)
    
        writer.add_scalars('MSE_epoch', mse_per_epoch, epoch)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal epochs! Finish training') 

## train_model
## ----------------------------------------------------------------------------

## ----------------------------------------------------------------------------
## main fun

# net = VDN.VDNU_Invert(args.chn+args.chn_forward_op, args.chn*2, args.chn, args.chn*2, \
#                       activation=args.activation, act_init=args.relu_init, \
#                       wf=args.wf, batch_norm=args.bn_UNet)

net = VDN.VDNU_Invert(args.chn, args.chn*2, args.chn, args.chn*2, \
                      activation=args.activation, act_init=args.relu_init, \
                      wf=args.wf, batch_norm=args.bn_UNet, \
                      u_dim=args.patch_size, dep_U=4, dep_S=5)
    
#net = nn.DataParallel(net).cuda()
net = net.cuda()

# optimizer
optimizer = optim.Adam(net.parameters(), lr=args.lr)
print('\nStepLR with gamma={:.2f}, step size={:d}'.format(args.gamma, args.step_size))
scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)

# if args.resume:
#     if os.path.isfile(args.resume):
#         print('=> Loading checkpoint {:s}'.format(args.resume))
#         checkpoint = torch.load(args.resume)
#         args.epoch_start = checkpoint['epoch']
#         args.step = checkpoint['step']
#         args.step_img = checkpoint['step_img']
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
#         net.load_state_dict(checkpoint['model_state_dict'])
#         args.clip_grad_D = checkpoint['grad_norm_D']
#         args.clip_grad_S = checkpoint['grad_norm_S']
#         print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args.resume, checkpoint['epoch']))
#     else:
#         sys.exit('Please provide corrected model path!')
# else:
#     args.epoch_start = 0
#     if os.path.isdir(args.log_dir):
#         shutil.rmtree(args.log_dir)
#     os.makedirs(args.log_dir)
#     if os.path.isdir(args.model_dir):
#         shutil.rmtree(args.model_dir)
#     os.makedirs(args.model_dir)

# print the arg pamameters
for arg in vars(args):
    print('{:<15s}: {:s}'.format(arg,  str(getattr(args, arg))))

# making traing data
simulate_dir = Path(args.simulate_dir) 
train_u_list = list((simulate_dir / 'u').glob('*.npy'))
train_d_list = list((simulate_dir / 'd').glob('*.npy'))
# sort the train_u_list and train_d_list to make sure that they are listed in 
# the same order
train_u_list = sorted([str(x) for x in train_u_list])
train_d_list = sorted([str(x) for x in train_d_list])

test_u_list = ['data/u_128.npy']

fun_size = 128
# matrix_HTH = np.load((args.data_dir + '/decompose_HTH/matrix_HTH.npy'))
# ch_f, _, _ = matrix_HTH.shape
# if args.chn_forward_op <= ch_f:
#     matrix_HTH = matrix_HTH[:args.chn_forward_op, :, :]
# else:
#     sys.exit("Channels of the forward operator are set too large!")

matrix_HTH = None

# for testing
# datasets = {'train': InvertingDatasets.SimulateTrain(train_u_list, train_u_list, \
#                      HTH=None, length=500*args.batch_size, radius=5, noise_type=args.noise,\
#                      noise_estimate=True, chn=args.chn)}

args.noise = 'iidgauss'
args.print_freq = 100
datasets = {'train': InvertingDatasets.SimulateTrain(train_u_list, train_d_list, \
                      HTH=matrix_HTH, length=500*args.batch_size, radius=5, noise_type=args.noise,\
                      noise_estimate=True, chn=args.chn)}

# train model
print('\nBegin training with GPU: ' + str(args.gpu_id))
if args.learn_type == 'residual':
    loss_fn = loss_fn_residual 
elif args.learn_type == 'clean':
    loss_fn = loss_fn_clean  
else:
    sys.exit("loss_fn_type must be residual or clean")


## ------------------------------------------------------
log_dir = args.log_dir + '_VIRN'
model_dir = args.log_dir = '_VIRN'
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)
if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir)

args.epoch_start = 0

_lr_min = 1e-7
optimizer = optim.Adam(net.parameters(), lr=1e-4)
print('\nStepLR with gamma={:.2f}, step size={:d}'.format(args.gamma, args.step_size))
scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
train_model(net, datasets, optimizer, scheduler, loss_fn, \
            forward_op=forwardOP_full.forward_op, epoch_num=100, \
            log_dir=log_dir, model_dir=model_dir)   
print("All Net training completed!")   

model_state_prefix = 'model_state'
save_path_model_state = os.path.join(args.model_dir, model_state_prefix)
torch.save(net.state_dict(), save_path_model_state) 

## main fun
## ----------------------------------------------------------------------------



















