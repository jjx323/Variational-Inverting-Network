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
from tensorboardX import SummaryWriter
from math import ceil
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
from pathlib import Path
import fenics as fe
from scipy import sparse
# import h5py as h5
import cv2
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

import matplotlib.pyplot as plt

sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
from VarInvNetHelmholtzSource.utils import batch_PSNR, batch_SSIM
from VarInvNetHelmholtzSource.loss import loss_fn_residual, loss_fn_clean, loss_fn_clean_FEM
from VarInvNetHelmholtzSource.networks import VDN
from VarInvNetHelmholtzSource.datasets import InvertingDatasets
from VarInvNetHelmholtzSource.options import set_opts
from core.model import Domain2D
from core.misc import load_expre
from core.misc import trans_to_python_sparse_matrix
from core.probability import GaussianMeasure, Noise
from VarInvNetHelmholtzSource.common import EquSolver
from Auxillary import gene_vec2matrix, gene_matrix2vec, ForwardOP, \
                      ForwardOP_Rough, GenerateMf


# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

args = set_opts()
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# if isinstance(args.gpu_id, int):
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
# else:
#     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

_lr_min = 1e-7
_modes = ['train', 'test']

# _lr_min = 1e-8
# args.lr = 1e-6
args.batch_size = 16
args.learn_type = 'clean'
args.chn = 1
args.patch_size = 160
args.repeat_number = 4

mesh = fe.Mesh('./data_train_test/saved_mesh.xml')
V = fe.FunctionSpace(mesh, 'CG', 1)
fun = fe.Function(V)
# args.patch_size = np.int(np.sqrt(len(fun.vector()[:]))) + 1
## ----------------------------------------------------------------------------
## set the forward operator

data_equ_dir = './data_train_test/classical_method_data'

## domain for solving PDE

coordinates = np.load(data_equ_dir + '/coordinates.npy')
freqs = np.load(data_equ_dir + '/freqs.npy')
index_freq = list(np.arange(len(freqs)))
# np.random.shuffle(index_freq)
index_freq = sorted(index_freq[:48])  # only use 24 wavenumbers
# index_freq = sorted(np.random.choice(len(freqs), 24, replace=False))
# index_freq = list(np.arange(30))[6:]
# index_freq[0], index_freq[1], index_freq[2], index_freq[3] = 0, 2, 4, 8
np.save('index_freq', index_freq)
freqs = freqs[index_freq]

mesh_size = args.patch_size
forwardOP_full = ForwardOP(mesh_size=mesh_size, f=fe.Constant("1.0"), \
                           coordinates=coordinates, kappas=freqs)
   
# tau = 1#1/(noise_level**2)
# noise = Noise(dim=len(coordinates))
# noise.set_parameters_Gaussian(covariance_val=1/tau, eval_precision=True)
    
forwardOP_rough = ForwardOP_Rough(mesh_size=40, f=fe.Constant("1.0"), \
                           coordinates=coordinates, kappas=freqs)
# forwardOP_rough.eva_inv(data_equ_dir)
forwardOP_rough.load_inv(data_equ_dir)
print("inv matrix loaded")
forwardOP_rough.to_cuda()
print("all variables are on gpu")

geneMf = GenerateMf(forwardOP_full.V)
matrix2vec = gene_matrix2vec(forwardOP_full.V, device='gpu')
vec2matrix = gene_vec2matrix(forwardOP_full.V, device='gpu')
mesh = forwardOP_full.V.mesh()
mesh_coordinates = mesh.coordinates()
dx = (mesh_coordinates[1,0] - mesh_coordinates[0,0])
print("Preparations of the forward operator are completed!")

## set the forward operator
## ----------------------------------------------------------------------------

## ----------------------------------------------------------------------------
## train_model

def train_model(net, datasets, optimizer, lr_scheduler, criterion, train_type='train_all',\
                forward_op=None, epoch_num=args.epochs, log_dir=args.log_dir,  \
                model_dir=args.model_dir, index=index_freq):
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
    param_I = [x for name, x in net.named_parameters() if 'inet' in name.lower()]
    param_S = [x for name, x in net.named_parameters() if 'snet' in name.lower()]
    
    for epoch in range(args.epoch_start, epoch_num):
        loss_per_epoch = {x: 0 for x in ['Loss', 'lh', 'KLG', 'KLIG']}
        mse_per_epoch = {x: 0 for x in _modes}
        grad_norm_D = grad_norm_S = grad_norm_I = 0
        tic = time.time()
        # train stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        if lr < _lr_min:
            sys.exit('Reach the minimal learning rate')
        phase = 'train'
        
        for ii, data in enumerate(data_loader[phase]):
            # start = time.time()
            d_noise, d_clean, u, sigmaMapEst, sigmaMapGt = [x.cuda() for x in data]
            # end = time.time()
            # print('data loade', end-start)
            # print(d_noise.shape)
            
            # start = time.time()
            optimizer.zero_grad()
            if train_type == 'train_INet' or train_type == 'train_all':
                phi_Z, phi_sigma, est_u = net(d_noise, train_type)
                _, est_u_num, _, _ = est_u.shape
                est_u = est_u[:, est_u_num-3, :, :][:, np.newaxis, :, :]
                d_noise = d_noise[:,:,0:-1:args.repeat_number,:]
                d_clean = d_clean[:,:,0:-1:args.repeat_number,:]
                sigmaMapEst = sigmaMapEst[:,:,0:-1:args.repeat_number,:]
                sigmaMapGt = sigmaMapGt[:,:,0:-1:args.repeat_number,:]
            else:
                phi_Z, phi_sigma = net(d_noise, train_type)
                # print(phi_sigma.shape)
            # end = time.time()
            # print('forward computed', end-start)

            # start = time.time()
            if train_type == 'train_DNet_SNet':
                # only learn the denoising UNet, the noisy data is d_noise, 
                # the clean data is d_clean
                forward_op = None 
                loss, g_lh, kl_g, kl_Igam = criterion(phi_Z, phi_sigma, d_noise, d_clean, \
                                                      sigmaMapEst, args.eps2, \
                                                      dx=dx, radius=args.radius,\
                                                      forward_op=forward_op, \
                                                      kl_gauss_type='full', geneMf=geneMf, \
                                                      vec2matrix=vec2matrix, matrix2vec=matrix2vec)
            elif train_type == 'train_INet':
                loss, g_lh, kl_g, kl_Igam = criterion(phi_Z, phi_sigma, d_noise, u, sigmaMapEst, \
                                                      args.eps2, radius=args.radius,\
                                                      forward_op=forward_op, dx=dx, \
                                                      kl_gauss_type='MC', geneMf=geneMf, \
                                                    vec2matrix=vec2matrix, matrix2vec=matrix2vec)
            elif train_type == 'train_all':
                loss, g_lh, kl_g, kl_Igam = criterion(phi_Z, phi_sigma, d_noise, u, sigmaMapEst, \
                                                      args.eps2, radius=args.radius,\
                                                      forward_op=forward_op, dx=dx, \
                                                      kl_gauss_type='MC', geneMf=geneMf,  \
                                                    vec2matrix=vec2matrix, matrix2vec=matrix2vec)
            else:
                sys.exit("train_type must be in ['train_DNet_SNet', 'train_INet', 'train_all']")
            # end = time.time()
            # print('loss calculated', end-start)
            
            # start = time.time()
            loss.backward()
            # clip the gradnorm
            total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)
            total_norm_I = nn.utils.clip_grad_norm_(param_I, clip_grad_I)
            total_norm_S = nn.utils.clip_grad_norm_(param_S, clip_grad_S)
            grad_norm_D = (grad_norm_D*(ii/(ii+1)) + total_norm_D/(ii+1))
            grad_norm_I = (grad_norm_I*(ii/(ii+1)) + total_norm_I/(ii+1))
            grad_norm_S = (grad_norm_S*(ii/(ii+1)) + total_norm_S/(ii+1))
            optimizer.step()
            # end = time.time()
            # print('gradient computed', end-start)
            
            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            loss_per_epoch['lh'] += g_lh.item() / num_iter_epoch[phase]
            loss_per_epoch['KLG'] += kl_g.item() / num_iter_epoch[phase]
            loss_per_epoch['KLIG'] += kl_Igam.item() / num_iter_epoch[phase]
            
            if train_type == 'train_DNet_SNet':
                d_invert = phi_Z[:, :C,].detach().data
                mse = F.mse_loss(d_invert, d_clean)
                mse_per_epoch[phase] += mse
            elif train_type == 'train_INet' or train_type == 'train_all':
                u_invert = phi_Z[:, :C,].detach().data
                mse = F.mse_loss(u_invert, u)
                mse_per_epoch[phase] += mse
                mse_c = F.mse_loss(est_u, u)
            else:
                u_invert = phi_Z[:, :C,].detach().data
                mse = F.mse_loss(u_invert, u)
                mse_per_epoch[phase] += mse
            if (ii+1)%args.print_freq == 0:
                if train_type == 'train_INet' or train_type == 'train_all':
                    log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, lh={:+4.2f}, ' + \
                                'KLG={:+>7.2f}, KLIG={:+>6.2f}, mse={:.2e}, mse_c={:.2e}, ' + \
                                    'GNorm_D:{:.1e}/{:.1e}, GNorm_I:{:.1e}/{:.1e}, ' + \
                                    'GNorm_S:{:.1e}/{:.1e}, lr={:.1e}'
                    print(log_str.format(epoch+1, args.epochs, phase, ii+1, num_iter_epoch[phase], \
                                          g_lh.item(), kl_g.item(), kl_Igam.item(), mse, mse_c, \
                                          clip_grad_D, total_norm_D, clip_grad_I, total_norm_I, \
                                          clip_grad_S, total_norm_S, lr))
                else:
                    log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, lh={:+4.2f}, ' + \
                                    'KLG={:+>7.2f}, KLIG={:+>6.2f}, mse={:.2e},' + \
                                        'GNorm_D:{:.1e}/{:.1e}, GNorm_I:{:.1e}/{:.1e}, ' + \
                                        'GNorm_S:{:.1e}/{:.1e}, lr={:.1e}'
                    print(log_str.format(epoch+1, args.epochs, phase, ii+1, num_iter_epoch[phase], \
                                              g_lh.item(), kl_g.item(), kl_Igam.item(), mse, \
                                              clip_grad_D, total_norm_D, clip_grad_I, total_norm_I, \
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
                if train_type == 'train_DNet_SNet':
                    x1 = vutils.make_grid(d_invert, normalize=True, scale_each=True)
                    writer.add_image(phase+' Estimated data d', x1, step_fun[phase])
                else:
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
                if train_type == 'train_INet' or train_type == 'train_all':
                    x6 = vutils.make_grid(est_u, normalize=True, scale_each=True)
                    writer.add_image(phase+' u obtained by regularization', x6, step_fun[phase])
                step_fun[phase] += 1
        
        mse_per_epoch[phase] /= (ii+1)
        log_str ='{:s}: Loss={:+.2e}, lh={:+.2e}, KL_Guass={:+.2e}, KLIG={:+.2e}, mse={:.3e}, ' + \
                  'GNorm_D={:.1e}/{:.1e}, GNorm_I={:.1e}/{:.1e}, GNorm_S={:.1e}/{:.1e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['lh'],
                                loss_per_epoch['KLG'], loss_per_epoch['KLIG'], mse_per_epoch[phase],
                                clip_grad_D, grad_norm_D, clip_grad_I, grad_norm_I, \
                                clip_grad_S, grad_norm_S))
        writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
        writer.add_scalar('Mean Grad Norm_D epoch', grad_norm_D, epoch)
        writer.add_scalar('Mean Grad Norm_I epoch', grad_norm_I, epoch)
        writer.add_scalar('Mean Grad Norm_S epoch', grad_norm_S, epoch)
        clip_grad_D = min(clip_grad_D, grad_norm_D)
        clip_grad_I = min(clip_grad_I, grad_norm_I)
        clip_grad_S = min(clip_grad_S, grad_norm_S)
        print('-'*150)
        
        # test stage
        net.eval()
        phase = 'test'
        # d_clean = np.load('./data/u_128.npy')  # just for testing
        # d_clean = np.load('./data/d_clean.npy')
        # d_len = np.int(np.sqrt(len(d_clean)))
        # # d_len, _ = d_clean.shape
        # d_clean = d_clean.reshape(d_len, d_len)
        u = np.load('./data_train_test/test_data_u160/u_0.npy')
        # u = np.load('./data_train_test/u_classic.npy')
        # u = np.load('./data_train_test/u_union.npy')
        lu, ru = u.shape
        
        if train_type == 'train_DNet_SNet':
            repeat_number = args.repeat_number
        else:
            repeat_number = args.repeat_number
        d_clean = np.load('./data_train_test/test_data_dr/dr_0.npy')
        # d_clean = np.load('./data_train_test/dr_classic.npy')
        # d_clean = np.load('./data_train_test/dr_union.npy')
        d_clean = d_clean[index, :]
        noise_level = 0.05
        d_noise = np.load('./data_train_test/test_data_d/d_0.npy')
        # d_noise = np.load('./data_train_test/d_classic.npy')
        # d_noise = np.load('./data_train_test/d_union.npy')
        d_noise = d_noise[index, :]
        dl, dr = d_noise.shape
        d_noise = d_noise + noise_level*\
            np.repeat(np.max(np.abs(d_clean), axis=1).reshape(dl,1), dr, axis=1)\
                    *np.random.randn(dl, dr)
        # d_noise = cv2.resize(d_noise, (lu, ru), interpolation=cv2.INTER_AREA)[np.newaxis, :, :]        
        u = torch.tensor(u[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        
        d_clean = np.repeat(d_clean, repeat_number, axis=0)
        d_noise = np.repeat(d_noise, repeat_number, axis=0)
        
        d_noise = torch.tensor(d_noise[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        d_clean = torch.tensor(d_clean[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        d_noise, d_clean, u = d_noise.cuda(), d_clean.cuda(), u.cuda()
        with torch.set_grad_enabled(False):
            if train_type == 'train_INet' or train_type == 'train_all':
                phi_Z, phi_sigma, est_u = net(d_noise, train_type)
                _, est_u_num, _, _ = est_u.shape
                est_u = est_u[:, est_u_num-3, :, :][:, np.newaxis, :, :]
            else:
                phi_Z, phi_sigma = net(d_noise, train_type)

        
        if train_type == 'train_DNet_SNet':
            d_invert = phi_Z[:, :C,].detach().data
            mse = F.mse_loss(d_invert, d_clean)
            print("Test: mse = {:.2e}".format(mse))
            print("noise clean mse = {:.2e}".format(F.mse_loss(d_noise, d_clean)))
        elif train_type == 'train_INet' or train_type == 'train_all':
            u_invert = phi_Z[:, :C,].detach().data
            mse = F.mse_loss(u_invert, u)
            mse_c = F.mse_loss(est_u, u)
            print("Test: mse = {:.2e}, mse_c = {:.2e}".format(mse, mse_c))
        else:
            u_invert = phi_Z[:, :C,].detach().data
            mse = F.mse_loss(u_invert, u)
            print("Test: mse = {:.2e}".format(mse))
        
        # tensorboardX summary
        alpha = torch.exp(phi_sigma[:, :C, ])
        beta = torch.exp(phi_sigma[:, C:, ])
        sigmaMap_pred = beta / (alpha-1)
        if train_type == 'train_DNet_SNet':
            x1 = vutils.make_grid(d_invert, normalize=True, scale_each=True)
            writer.add_image(phase+' Denoised data', x1, step_fun[phase])
        else:
            x1 = vutils.make_grid(u_invert, normalize=True, scale_each=True)
            writer.add_image(phase+' Inverted u', x1, step_fun[phase])
        x2 = vutils.make_grid(u, normalize=True, scale_each=True)
        writer.add_image(phase+' GroundTruth', x2, step_fun[phase])
        x3 = vutils.make_grid(sigmaMap_pred, normalize=True, scale_each=True)
        writer.add_image(phase+' Predict Sigma', x3, step_fun[phase])
        x4 = vutils.make_grid(d_noise, normalize=True, scale_each=True)
        writer.add_image(phase+' Noise data', x4, step_fun[phase])
        x44 = vutils.make_grid(d_clean, normalize=True, scale_each=True)
        writer.add_image(phase+' Clean data', x44, step_fun[phase])
        if train_type == 'train_INet' or train_type == 'train_all':
            x5 = vutils.make_grid(est_u, normalize=True, scale_each=True)
            writer.add_image(phase+' u obtained by regularization', x5, step_fun[phase])
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

num_freqs = len(freqs)
data_dim = len(coordinates)

freq_used = [10, 20, 30, 40, 47]  # this specifies the index of the wavenumber 
# freq_used = [4, 9, 14, 19, 23]  # this specifies the index of the wavenumber 

net = VDN.VDNU_Invert_Null(in_chn_U=args.chn, out_chn_U=args.chn*2, \
                           in_chn_S=args.chn, out_chn_S=args.chn*2, \
                           in_chn_I=len(freq_used)+2, out_chn_I=args.chn*2, \
                      activation=args.activation, act_init=args.relu_init, \
                      wf=args.wf, batch_norm=args.bn_UNet, forward_op=forwardOP_rough,\
                      data_dim=data_dim, u_dim=args.patch_size, dep_U=4, dep_I=4, dep_S=5,\
                          repeat_number=args.repeat_number, freq_used=freq_used)
    
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
# u_dir = 'samples_u' + str(args.patch_size)
# train_u_list = list((simulate_dir / u_dir).glob('*.npy'))
# train_d_list = list((simulate_dir / 'samples_d').glob('*.npy'))
# train_dr_list = list((simulate_dir / 'samples_dr').glob('*.npy'))
# # sort the train_u_list and train_d_list to make sure that they are listed in 
# # the same order
# train_u_list = sorted([str(x) for x in train_u_list])
# train_d_list = sorted([str(x) for x in train_d_list])
# train_dr_list = sorted([str(x) for x in train_dr_list])

training_data_list = list((simulate_dir / 'training_data').glob('*.npy'))

test_u_list = ['./data_train_test/test_data_u/u_0.npy']

# fun_size = 400

args.noise = 'iidgauss'
args.print_freq = 100
datasets = {'train': InvertingDatasets.SimulateTrain(training_data_list, \
                      length=500*args.batch_size, radius=5, noise_type=args.noise,\
                      noise_estimate=True, chn=args.chn, index=index_freq, \
                      repeat_number=args.repeat_number)}

# train model
print('\nBegin training with GPU: ' + str(args.gpu_id))
if args.learn_type == 'residual':
    loss_fn = loss_fn_residual 
elif args.learn_type == 'clean':
    loss_fn = loss_fn_clean_FEM 
else:
    sys.exit("loss_fn_type must be residual or clean")

## ------------------------------------------------------
train_d_noise = True
if train_d_noise == True:
    if os.path.isdir(args.log_dir + '/log_D'):
        shutil.rmtree(args.log_dir + '/log_D')
    os.makedirs(args.log_dir + '/log_D')
    if os.path.isdir(args.model_dir + '/model_D'):
        shutil.rmtree(args.model_dir + '/model_D')
    os.makedirs(args.model_dir + '/model_D')
    
    ## adjust the power of the prior
    args.eps2 = 1e-9
    args.epoch_start = 0
    args.radius = 3
    
    _lr_min = 1e-10
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    print('\nStepLR with gamma={:.2f}, step size={:d}'.format(args.gamma, args.step_size))
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    
    train_model(net, datasets, optimizer, scheduler, loss_fn, train_type='train_DNet_SNet', \
                forward_op=None, epoch_num=100, log_dir=args.log_dir+'/log_D', \
                model_dir=args.model_dir+'/model_D', index=index_freq)
    print("DNet and SNet training completed!")

model_state_prefix = 'model_state'
save_path_model_state = os.path.join(args.model_dir + '/model_D', model_state_prefix)
if train_d_noise == True:
    torch.save(net.state_dict(), save_path_model_state)
net.load_state_dict(torch.load(save_path_model_state))

# # ## ------------------------------------------------------

train_inet = True
    
if train_inet == True:
    if os.path.isdir(args.log_dir + '/log_I'):
        shutil.rmtree(args.log_dir + '/log_I')
    os.makedirs(args.log_dir + '/log_I')
    if os.path.isdir(args.model_dir + '/model_I'):
        shutil.rmtree(args.model_dir + '/model_I')
    os.makedirs(args.model_dir + '/model_I')
    
    args.eps2 = 1e-6
    args.epoch_start = 0
    args.radius = 5
    
    _lr_min = 1e-8
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    print('\nStepLR with gamma={:.2f}, step size={:d}'.format(args.gamma, args.step_size))
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    
    train_model(net, datasets, optimizer, scheduler, loss_fn, train_type='train_INet', \
                forward_op=forwardOP_full.forward_op, epoch_num=100, \
                log_dir=args.log_dir+'/log_I', model_dir=args.model_dir+'/model_I', 
                index=index_freq)   
    print("INet training completed!")   
    
    model_state_prefix = 'model_state'
    save_path_model_state = os.path.join(args.model_dir + '/model_I', model_state_prefix)
    torch.save(net.state_dict(), save_path_model_state) 
    net.load_state_dict(torch.load(save_path_model_state))

## ------------------------------------------------------
# if os.path.isdir(args.log_dir + '/log_all'):
#     shutil.rmtree(args.log_dir + '/log_all')
# os.makedirs(args.log_dir + '/log_all')
# if os.path.isdir(args.model_dir + '/model_all'):
#     shutil.rmtree(args.model_dir + '/model_all')
# os.makedirs(args.model_dir + '/model_all')

# args.eps2 = 1e-5
# args.epoch_start = 0
# args.radius = 5

# _lr_min = 1e-8
# optimizer = optim.Adam(net.parameters(), lr=1e-4)
# print('\nStepLR with gamma={:.2f}, step size={:d}'.format(args.gamma, args.step_size))
# scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
# train_model(net, datasets, optimizer, scheduler, loss_fn, train_type='train_all', \
#             forward_op=forwardOP_full.forward_op, epoch_num=5, \
#             log_dir='./log' + '/log_all', model_dir=args.model_dir+'/model_all', \
#             index=index_freq)   
# print("All Net training completed!")   

# model_state_prefix = 'model_state'
# save_path_model_state = os.path.join(args.model_dir + '/model_all', model_state_prefix)
# torch.save(net.state_dict(), save_path_model_state) 

## main fun
## ----------------------------------------------------------------------------



















