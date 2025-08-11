#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:37:55 2021

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
from VarInvNetHelmholtzSource.common import EquSolverPML, DomainPML
from Auxillary import gene_vec2matrix, gene_matrix2vec, ForwardOP, \
                      ForwardOP_Rough, GenerateMf


# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

args = set_opts()

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

##-------------------------------------------------------------------------------
## VINet
##-------------------------------------------------------------------------------
num_freqs = len(freqs)
data_dim = len(coordinates)

freq_used = [10, 20, 30, 40, 47]

net = VDN.VDNU_Invert_Null(in_chn_U=args.chn, out_chn_U=args.chn*2, \
                            in_chn_S=args.chn, out_chn_S=args.chn*2, \
                            in_chn_I=len(freq_used)+2, out_chn_I=args.chn*2, \
                      activation=args.activation, act_init=args.relu_init, \
                      wf=args.wf, batch_norm=args.bn_UNet, forward_op=forwardOP_rough,\
                      data_dim=data_dim, u_dim=args.patch_size, dep_U=4, dep_I=4, dep_S=5,\
                          repeat_number=args.repeat_number, freq_used=freq_used)

model_state_prefix = 'model_state'
save_path_model_state = os.path.join(args.model_dir + '/model_I', model_state_prefix)
net.load_state_dict(torch.load(save_path_model_state))
# net = net.cuda()

###----------------------------------------------------------------------------
## VINet  d_union
###----------------------------------------------------------------------------

net.eval()
ut = np.load('./data_train_test/samples2_u160/u2_0.npy')
u = ut.copy()
lu, ru = u.shape

repeat_number = args.repeat_number

C = 1
ii = 0
index = index_freq
# d_clean = np.load('./data_train_test/test_data_dr/dr_' + str(ii) + '.npy')
# d_clean = np.load('./data_train_test/dr_union.npy')
# d_clean = d_clean[index, :]
# d_clean = np.repeat(d_clean, repeat_number, axis=0)
noise_level = 0.05
d_noise = np.load('./data_train_test/samples_d2/d2_0.npy')
d_noise = d_noise[index, :]
dl, dr = d_noise.shape
d_noise = d_noise + noise_level*\
    np.repeat(np.max(np.abs(d_noise), axis=1).reshape(dl,1), dr, axis=1)\
            *np.random.randn(dl, dr)
d_noise = np.repeat(d_noise, repeat_number, axis=0)
# d_noise = cv2.resize(d_noise, (lu, ru), interpolation=cv2.INTER_AREA)[np.newaxis, :, :]        
u = torch.tensor(u[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
   

d_noise = torch.tensor(d_noise[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
# d_clean = torch.tensor(d_clean[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
# d_noise, d_clean, u = d_noise.cuda(), d_clean.cuda(), u.cuda()

phi_Z, phi_sigma, est_u = net(d_noise, 'train_inet')
u_invert = phi_Z[:, :C,].detach().data

fest_vinet1 = fe.Function(forwardOP_full.domainPML.VR)
ftrue_1 = fe.Function(forwardOP_full.domainPML.VR)
fest_vinet1.vector()[:] = matrix2vec(u_invert)[0,0,:]
ftrue_1.vector()[:] = matrix2vec(ut[np.newaxis, np.newaxis, :, :])[0,0,:]
fenzi = fe.assemble(fe.inner(fest_vinet1-ftrue_1, fest_vinet1-ftrue_1)*fe.dx)
fenmu = fe.assemble(fe.inner(ftrue_1, ftrue_1)*fe.dx)
relative_error1 = fenzi/fenmu

## ------------------------------------------------------------------------
net.eval()
ut = np.load('./data_train_test/test_data_u160/u_0.npy')
u = ut.copy()
lu, ru = u.shape

repeat_number = args.repeat_number

C = 1
ii = 0
index = index_freq
# d_clean = np.load('./data_train_test/test_data_dr/dr_' + str(ii) + '.npy')
# d_clean = np.load('./data_train_test/dr_union.npy')
# d_clean = d_clean[index, :]
# d_clean = np.repeat(d_clean, repeat_number, axis=0)
noise_level = 0.05
d_noise = np.load('./data_train_test/test_data_d/d_0.npy')
d_noise = d_noise[index, :]
dl, dr = d_noise.shape
d_noise = d_noise + noise_level*\
    np.repeat(np.max(np.abs(d_noise), axis=1).reshape(dl,1), dr, axis=1)\
            *np.random.randn(dl, dr)
d_noise = np.repeat(d_noise, repeat_number, axis=0)
# d_noise = cv2.resize(d_noise, (lu, ru), interpolation=cv2.INTER_AREA)[np.newaxis, :, :]        
u = torch.tensor(u[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
   

d_noise = torch.tensor(d_noise[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
# d_clean = torch.tensor(d_clean[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
# d_noise, d_clean, u = d_noise.cuda(), d_clean.cuda(), u.cuda()

phi_Z, phi_sigma, est_u = net(d_noise, 'train_inet')
u_invert = phi_Z[:, :C,].detach().data

fest_vinet2 = fe.Function(forwardOP_full.domainPML.VR)
ftrue_2 = fe.Function(forwardOP_full.domainPML.VR)

fest_vinet2.vector()[:] = matrix2vec(u_invert)[0,0,:]
ftrue_2.vector()[:] = matrix2vec(ut[np.newaxis, np.newaxis, :, :])[0,0,:]
fenzi = fe.assemble(fe.inner(fest_vinet2-ftrue_2, fest_vinet2-ftrue_2)*fe.dx)
fenmu = fe.assemble(fe.inner(ftrue_2, ftrue_2)*fe.dx)
relative_error2 = fenzi/fenmu

###----------------------------------------------------------------------------
## classical RLM 
###----------------------------------------------------------------------------
## domain for solving PDE
nnn = 160-1
domain= DomainPML(nx=nnn, ny=nnn)
domainS = Domain2D(nx=nnn, ny=nnn, mesh_type='CG', mesh_order=1)

ftrue_2_RLM = fe.interpolate(ftrue_2, domain.VR)

f_init = fe.Constant("0.0")

noise_level = 0.05
d_noise = np.load('./data_train_test/test_data_d/d_0.npy')
d_noise = d_noise[index, :]
dl, dr = d_noise.shape
d_noise = d_noise + noise_level*\
    np.repeat(np.max(np.abs(d_noise), axis=1).reshape(dl,1), dr, axis=1)\
            *np.random.randn(dl, dr)

def relative_error(f, f_true):
    fenzi = fe.assemble(fe.inner(f-f_true, f-f_true)*fe.dx)
    fenmu = fe.assemble(fe.inner(f_true, f_true)*fe.dx)
    return fenzi/fenmu

def solve_RLM(domain, d_noise, f_init, coordinates, ftrue, freqs):
    f_iter = fe.Function(domain.VR)
    # f_iter_ = fe.Function(domain.VR)
    equ_solver = EquSolverPML(domain=domain, kappa=1, fR=f_init, points=coordinates)
    equ_solver.geneForwardNumpyMatrix()
    equ_solver.geneAdjointNumpyMatrix()
    
    start = time.time()
    for ii, freq in enumerate(freqs):
        equ_solver.update_k(freq)
        equ_solver.update_f(f_iter.vector()[:])
        equ_solver.forward_solve()
        data = equ_solver.get_data(equ_solver.forward_sol_vec)
        equ_solver.adjoint_solve(data - d_noise[ii, :])
        grad_val = equ_solver.adjoint_sol_vec[::2] 
        gnorm = np.max(np.abs(grad_val)) + 1e-15
        
        step_length = 0.05
            
        f_iter.vector()[:] += -step_length*grad_val/gnorm
        print("freq = {:.2f}, relative_error = {:.2f}, step_length = {:.5f}".format(freq, \
                                                      relative_error(f_iter, ftrue), step_length))
    
    end = time.time()
    print("Total Time: ", end-start)
    
    return f_iter 
    
fest_rlm2 = solve_RLM(domain, d_noise, f_init, coordinates, ftrue_2_RLM, freqs)

## ---------------------------------------------------------------------------------

ftrue_1_RLM = fe.interpolate(ftrue_1, domain.VR)

f_init = fe.Constant("0.0")

noise_level = 0.05
d_noise = np.load('./data_train_test/samples_d2/d2_0.npy')
d_noise = d_noise[index, :]
dl, dr = d_noise.shape
d_noise = d_noise + noise_level*\
    np.repeat(np.max(np.abs(d_noise), axis=1).reshape(dl,1), dr, axis=1)\
            *np.random.randn(dl, dr)
    
fest_rlm1 = solve_RLM(domain, d_noise, f_init, coordinates, ftrue_1_RLM, freqs)


## ---------------------------------------------------------------------------------
##  calculate by RLM with 60 number of wavenumbers
## ---------------------------------------------------------------------------------

num_freqs = 60
freqs60 = np.linspace(1, 60, num_freqs)

f_init = fe.Constant("0.0")

noise_level = 0.05
d_noise = np.load('./data_train_test/samples_all/' + '/sample_60fre.npy')[:,:,0]
dl, dr = d_noise.shape
d_noise = d_noise + noise_level*\
    np.repeat(np.max(np.abs(d_noise), axis=1).reshape(dl,1), dr, axis=1)\
            *np.random.randn(dl, dr)
            
fest60_rlm2 = solve_RLM(domain, d_noise, f_init, coordinates, ftrue_2_RLM, freqs60)

## ---------------------------------------------------------------------------------

f_init = fe.Constant("0.0")

noise_level = 0.05
d_noise = np.load('./data_train_test/samples_all/' + '/sample2_60fre.npy')[:,:,0]
dl, dr = d_noise.shape
d_noise = d_noise + noise_level*\
    np.repeat(np.max(np.abs(d_noise), axis=1).reshape(dl,1), dr, axis=1)\
            *np.random.randn(dl, dr)
    
fest60_rlm1 = solve_RLM(domain, d_noise, f_init, coordinates, ftrue_1_RLM, freqs60)

##############################################################################
from Plotting import mplot_function

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 0.4
fig, axs = plt.subplots(nrows=1, ncols=5, \
      gridspec_kw={"width_ratios":[1,1,1,1,0.06], "height_ratios":[1]}, figsize=(20, 5))
# vim, vmax = -0.5, 0.4
vim, vmax = -.8, 0.
pad = 15
fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig1 = mplot_function(ftrue_2_RLM, axs[0], vmin=vim, vmax=vmax)
axs[0].set_title("(a) Background truth", pad=pad)
# axs[0,0].set_xlabel("$x$")
# axs[0,0].set_ylabel("$y$")
axs[0].set_xticks([0, 0.5, 1])
axs[0].set_yticks([0, 0.5, 1])
fig2 = mplot_function(fest_vinet2, axs[1], vmin=vim, vmax=vmax)
axs[1].set_title("(b) Estimate (VINet)", pad=pad)
# axs[0,1].set_xlabel("$x$")
# axs[0,1].set_ylabel("$y$")
axs[1].set_xticks([0, 0.5, 1])
axs[1].set_yticks([0, 0.5, 1])
fig3 = mplot_function(fest_rlm2, axs[2], vmin=vim, vmax=vmax)
axs[2].set_title("(c) Estimate (RLM with $d_{48}$)", pad=pad)
# axs[0,2].set_xlabel("$x$")
# axs[0,2].set_ylabel("$y$")
axs[2].set_xticks([0, 0.5, 1])
axs[2].set_yticks([0, 0.5, 1])
fig4 = mplot_function(fest60_rlm2, axs[3], vmin=vim, vmax=vmax)
axs[3].set_title("(d) Estimate (RLM with $d_{60}$)", pad=pad)
# axs[0,3].set_xlabel("$x$")
# axs[0,3].set_ylabel("$y$")
axs[3].set_xticks([0, 0.5, 1])
axs[3].set_yticks([0, 0.5, 1])
cbar = fig.colorbar(fig1, cax=axs[4], shrink=1)
cbar.ax.locator_params(nbins=5)
plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2)                     

## --------------------------------------------------------------------------
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 0.4
fig, axs = plt.subplots(nrows=1, ncols=5, \
      gridspec_kw={"width_ratios":[1,1,1,1,0.06], "height_ratios":[1]}, figsize=(20, 5))
# vim, vmax = -0.5, 0.4
vim, vmax = -0.6, 0.5
pad = 15
fig.subplots_adjust(wspace=0.35, hspace=0.3)
fig1 = mplot_function(ftrue_1_RLM, axs[0], vmin=vim, vmax=vmax)
axs[0].set_title("(a) Background truth", pad=pad)
# axs[0,0].set_xlabel("$x$")
# axs[0,0].set_ylabel("$y$")
axs[0].set_xticks([0, 0.5, 1])
axs[0].set_yticks([0, 0.5, 1])
fig2 = mplot_function(fest_vinet1, axs[1], vmin=vim, vmax=vmax)
axs[1].set_title("(b) Estimate (VINet)", pad=pad)
# axs[0,1].set_xlabel("$x$")
# axs[0,1].set_ylabel("$y$")
axs[1].set_xticks([0, 0.5, 1])
axs[1].set_yticks([0, 0.5, 1])
fig3 = mplot_function(fest_rlm1, axs[2], vmin=vim, vmax=vmax)
axs[2].set_title("(c) Estimate (RLM with $d_{48}$)", pad=pad)
# axs[0,2].set_xlabel("$x$")
# axs[0,2].set_ylabel("$y$")
axs[2].set_xticks([0, 0.5, 1])
axs[2].set_yticks([0, 0.5, 1])
fig4 = mplot_function(fest60_rlm1, axs[3], vmin=vim, vmax=vmax)
axs[3].set_title("(d) Estimate (RLM with $d_{60}$)", pad=pad)
# axs[0,3].set_xlabel("$x$")
# axs[0,3].set_ylabel("$y$")
axs[3].set_xticks([0, 0.5, 1])
axs[3].set_yticks([0, 0.5, 1])
cbar = fig.colorbar(fig1, cax=axs[4], shrink=1)
cbar.ax.locator_params(nbins=5)
plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2)    














