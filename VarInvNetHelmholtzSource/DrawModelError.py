#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:52:56 2021

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
import matplotlib as mpl

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
# args.batch_size = 32
args.learn_type = 'clean'
args.chn = 1
args.patch_size = 160
args.radius = 3
args.repeat_number = 4

mesh = fe.Mesh('./data_train_test/saved_mesh.xml')
V = fe.FunctionSpace(mesh, 'CG', 1)
fun = fe.Function(V)
# args.patch_size = np.int(np.sqrt(len(fun.vector()[:]))) + 1
## ----------------------------------------------------------------------------
## set the forward operator

data_equ_dir = './classical_method/data'

## domain for solving PDE

# g_expre = load_expre(data_equ_dir + '/g_expre.txt')
# u_expre = load_expre(data_equ_dir + '/u_expre.txt')
# g = fe.Expression(g_expre, degree=2)
# u = fe.Expression(u_expre, degree=2)

coordinates = np.load(data_equ_dir + '/coordinates.npy')
freqs = np.load(data_equ_dir + '/freqs.npy')
# index_freq = list(np.arange(len(freqs)))
# np.random.shuffle(index_freq)
# index_freq = sorted(index_freq[:24])
# index_freq = sorted(np.random.choice(len(freqs), 24, replace=False))
index_freq = np.load('index_freq.npy')
freqs = freqs[index_freq]

mesh_size = args.patch_size
forwardOP_full = ForwardOP(mesh_size=mesh_size, f=fe.Constant("1.0"), \
                           coordinates=coordinates, kappas=freqs)
   
tau = 1#1/(noise_level**2)
noise = Noise(dim=len(coordinates))
noise.set_parameters_Gaussian(covariance_val=1/tau, eval_precision=True)
    
forwardOP_rough = ForwardOP_Rough(mesh_size=40, f=fe.Constant("1.0"), \
                           coordinates=coordinates, kappas=freqs)
data_equ_dir = './data_train_test/classical_method_data'
forwardOP_rough.load_inv(data_equ_dir)
print("inv matrix loaded")
# forwardOP_rough.to_cuda()
forwardOP_rough.to_tensor()
# print("all variables are on gpu")

geneMf = GenerateMf(forwardOP_full.V)
matrix2vec = gene_matrix2vec(forwardOP_full.V, device='gpu')
vec2matrix = gene_vec2matrix(forwardOP_full.V, device='gpu')
mesh = forwardOP_full.V.mesh()
mesh_coordinates = mesh.coordinates()
dx = (mesh_coordinates[1,0] - mesh_coordinates[0,0])
print("Preparations of the forward operator are completed!")


num_freqs = len(freqs)
data_dim = len(coordinates)

freq_used = [4, 9, 14, 19, 23]

net = VDN.VDNU_Invert_Null(in_chn_U=args.chn, out_chn_U=args.chn*2, \
                           in_chn_S=args.chn, out_chn_S=args.chn*2, \
                           in_chn_I=len(freq_used)+2, out_chn_I=args.chn*2, \
                      activation=args.activation, act_init=args.relu_init, \
                      wf=args.wf, batch_norm=args.bn_UNet, forward_op=forwardOP_rough,\
                      data_dim=data_dim, u_dim=args.patch_size, dep_U=4, dep_I=4, dep_S=5,\
                          repeat_number=args.repeat_number, freq_used=freq_used)

model_state_prefix = 'model_state'
save_path_model_state = os.path.join(args.model_dir + '/model_D', model_state_prefix)
net.load_state_dict(torch.load(save_path_model_state))

net.eval()
ii = 0
# ut = np.load('./data_train_test/test_data_u160/u_0.npy')
# ut = np.load('./data_train_test/u_classic.npy')
# ut = np.load('./data_train_test/u_union.npy')
ut = np.load('./data_train_test/samples2_u160/u2_' + str(ii) + '.npy')
utt = np.load('./data_train_test/samples2_u160/u2_' + str(ii) + '.npy')
u = ut.copy()
lu, ru = u.shape

repeat_number = args.repeat_number

C = 1
index = index_freq
# d_clean = np.load('./data_train_test/test_data_dr/dr_0.npy')
# d_clean = np.load('./data_train_test/dr_classic.npy')
# d_clean = np.load('./data_train_test/dr_union.npy')
d_clean = np.load('./data_train_test/samples_dr2/dr2_' + str(ii) + '.npy')

d_clean = d_clean[index, :]
noise_level = 0.05
# d_noise = np.load('./data_train_test/test_data_d/d_0.npy')
# d_noise = np.load('./data_train_test/d_classic.npy')
# d_noise = np.load('./data_train_test/d_union.npy')
d_noise = np.load('./data_train_test/samples_d2/d2_' + str(ii) + '.npy')
d_noise = d_noise[index, :]
dl, dr = d_noise.shape
d_noise = d_noise + noise_level *\
    np.repeat(np.max(np.abs(d_clean), axis=1).reshape(dl, 1), dr, axis=1)\
    * np.random.randn(dl, dr)
# d_noise = cv2.resize(d_noise, (lu, ru), interpolation=cv2.INTER_AREA)[np.newaxis, :, :]
u = torch.tensor(u[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

d_clean = np.repeat(d_clean, repeat_number, axis=0)
d_noise = np.repeat(d_noise, repeat_number, axis=0)

d_noise = torch.tensor(
    d_noise[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
d_clean = torch.tensor(
    d_clean[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
# d_noise, d_clean, u = d_noise.cuda(), d_clean.cuda(), u.cuda()

phi_Z, phi_sigma = net(d_noise, 'train_DNet_SNet')

d_invert = phi_Z[:, :C, ].detach().data
mse = F.mse_loss(d_invert, d_clean)
print("Test: mse = {:.2e}".format(mse))
print("noise clean mse = {:.2e}".format(F.mse_loss(d_noise, d_clean)))


dc = d_clean.detach().cpu().numpy()[0, 0, 0:-1:repeat_number, :]
dn = d_noise.detach().cpu().numpy()[0, 0, 0:-1:repeat_number, :]
di = d_invert.detach().cpu().numpy()[0, 0, 0:-1:repeat_number, :]


def splitRI(x):
    ## split the real and imaginary part of the data
    l, r = x.shape
    xx = x.copy()
    r_half = np.int(r/2)
    # x[:, :r_half], x[:, r_half:] = x[:, ::2], x[:, 1::2]
    # return x
    xx[:, :r_half], xx[:, r_half:] = x[:, ::2], x[:, 1::2]
    return xx


def combineRI(x):
    l, r = x.shape
    xx = x.copy()
    rr = np.int(r/2)
    # for i in range(l):
    #     x[i, ::2], x[i, 1::2] = x[i, :rr], x[i, rr:]
    # return x
    for i in range(l):
        xx[i, ::2], xx[i, 1::2] = x[i, :rr], x[i, rr:]
    return xx

# dc1, dn, di = splitRI(dc), splitRI(dn), splitRI(di)
# dcc = combineRI(dc1)

dc, dn, di = splitRI(dc), splitRI(dn), splitRI(di)

fre1, fre2, fre3 = 10, 25, 45
loc=1
pad = 15
a1, a2 = 120, 160
xpart = list(np.arange(a1, a2))
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 0.4
plt.figure(figsize=(18, 12))
plt.subplot(2,3,1)
plt.plot(dc[fre1,:], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(dn[fre1,:], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(di[fre1,:], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(a) Comparison when $\kappa$ = " + str(fre1), pad=pad)
plt.subplot(2,3,2)
plt.plot(dc[fre2,:], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(dn[fre2,:], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(di[fre2,:], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(b) Comparison when $\kappa$ = " + str(fre2), pad=pad)
plt.subplot(2,3,3)
plt.plot(dc[fre3,:], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(dn[fre3,:], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(di[fre3,:], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(c) Comparison when $\kappa$ = " + str(fre3+1), pad=pad)
plt.subplot(2,3,4)
plt.plot(xpart, dc[fre1,a1:a2], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(xpart, dn[fre1,a1:a2], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(xpart, di[fre1,a1:a2], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(d) Comparison when $\kappa$ = " +str(fre1) + " (details)", pad=pad)
plt.subplot(2,3,5)
plt.plot(xpart, dc[fre2,a1:a2], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(xpart, dn[fre2,a1:a2], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(xpart, di[fre2,a1:a2], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(e) Comparison when $\kappa$ = " +str(fre2) + " (details)", pad=pad)
plt.subplot(2,3,6)
plt.plot(xpart, dc[fre3,a1:a2], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(xpart, dn[fre3,a1:a2], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(xpart, di[fre3,a1:a2], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(f) Comparison when $\kappa$ = " +str(fre3+1) + " (details)", pad=pad)
plt.tight_layout(pad=1, w_pad=1, h_pad=2)

###############################################################################
###############################################################################

net.eval()
ut = np.load('./data_train_test/test_data_u160/u_0.npy')
# ut = np.load('./data_train_test/u_classic.npy')
u = ut.copy()
lu, ru = u.shape

repeat_number = args.repeat_number

C = 1
index = index_freq
d_clean = np.load('./data_train_test/test_data_dr/dr_0.npy')
# d_clean = np.load('./data_train_test/dr_classic.npy')
d_clean = d_clean[index, :]
noise_level = 0.05
d_noise = np.load('./data_train_test/test_data_d/d_0.npy')
# d_noise = np.load('./data_train_test/d_classic.npy')
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
# d_noise, d_clean, u = d_noise.cuda(), d_clean.cuda(), u.cuda()

phi_Z, phi_sigma = net(d_noise, 'train_DNet_SNet')

d_invert = phi_Z[:, :C,].detach().data
mse = F.mse_loss(d_invert, d_clean)
print("Test: mse = {:.2e}".format(mse))
print("noise clean mse = {:.2e}".format(F.mse_loss(d_noise, d_clean)))


dc = d_clean.detach().cpu().numpy()[0,0,0:-1:repeat_number,:]
dn = d_noise.detach().cpu().numpy()[0,0,0:-1:repeat_number,:]
di = d_invert.detach().cpu().numpy()[0,0,0:-1:repeat_number,:]

dc, dn, di = splitRI(dc), splitRI(dn), splitRI(di)

fre1, fre2, fre3 = 10, 25, 45
loc=1
pad = 15
a1, a2 = 120, 160
xpart = list(np.arange(a1, a2))
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 0.4
plt.figure(figsize=(18, 12))
plt.subplot(2,3,1)
plt.plot(dc[fre1,:], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(dn[fre1,:], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(di[fre1,:], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(a) Comparison when $\kappa$ = " + str(fre1), pad=pad)
plt.subplot(2,3,2)
plt.plot(dc[fre2,:], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(dn[fre2,:], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(di[fre2,:], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(b) Comparison when $\kappa$ = " + str(fre2), pad=pad)
plt.subplot(2,3,3)
plt.plot(dc[fre3,:], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(dn[fre3,:], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(di[fre3,:], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(c) Comparison when $\kappa$ = " + str(fre3+1), pad=pad)
plt.subplot(2,3,4)
plt.plot(xpart, dc[fre1,a1:a2], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(xpart, dn[fre1,a1:a2], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(xpart, di[fre1,a1:a2], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(d) Comparison when $\kappa$ = " + str(fre1) + " (details)", pad=pad)
plt.subplot(2,3,5)
plt.plot(xpart, dc[fre2,a1:a2], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(xpart, dn[fre2,a1:a2], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(xpart, di[fre2,a1:a2], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(e) Comparison when $\kappa$ = " + str(fre2) + " (details)", pad=pad)
plt.subplot(2,3,6)
plt.plot(xpart, dc[fre3,a1:a2], label="clean data", color="dimgray", linewidth=3, linestyle="dotted")
plt.plot(xpart, dn[fre3,a1:a2], label="noisy data", color="red", linewidth=2, linestyle="dashed")
plt.plot(xpart, di[fre3,a1:a2], label="denoised data", color="darkblue", linewidth=2)
plt.legend(loc=loc)
plt.xlabel("parameters")
plt.ylabel("magnitude")
plt.title("(f) Comparison when $\kappa$ = " + str(fre3+1) + " (details)", pad=pad)
plt.tight_layout(pad=1, w_pad=1, h_pad=2)

###############################################################################
###############################################################################
u1 = np.load('./data_train_test/test_data_u160/u_0.npy')
# u2 = np.load('./data_train_test/u_classic.npy')
u2 = utt

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 17
plt.rcParams['axes.linewidth'] = 0.4
plt.figure(figsize=(13, 5))
pad = 20
plt.subplot(1, 2, 1)
ax = plt.gca()
label = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
plt.imshow(u1, origin='lower')
ax.set_xticks([np.int(a) for a in np.array(label)*160])
ax.set_xticklabels(label)
ax.set_yticks([np.int(a) for a in np.array(label)*160])
ax.set_yticklabels(label)
cbar = plt.colorbar()
cbar.ax.locator_params(nbins=6)
plt.title("(a) Source function generated by formula (3.11)", pad=pad)
plt.subplot(1, 2, 2)
ax = plt.gca()
plt.imshow(u2, origin='lower')
ax.set_xticks([np.int(a) for a in np.array(label)*160])
ax.set_xticklabels(label)
ax.set_yticks([np.int(a) for a in np.array(label)*160])
ax.set_yticklabels(label)
cbar = plt.colorbar()
cbar.ax.locator_params(nbins=5)
plt.title("(b) Source function generated by formula (3.12)", pad=pad)
plt.tight_layout(pad=1, w_pad=1, h_pad=2)
























