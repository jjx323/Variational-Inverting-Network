#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:41:25 2021

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
from loss import loss_fn_residual, loss_fn_clean, loss_fn_clean_FEM
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
# import h5py as h5
import cv2
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

import matplotlib.pyplot as plt

sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
from core.model import Domain2D
from core.misc import load_expre
from core.misc import trans_to_python_sparse_matrix
from core.probability import GaussianMeasure, Noise
from VarInvNetHelmholtzSource.common import EquSolver
from Auxillary import gene_vec2matrix, gene_matrix2vec, ForwardOP, ForwardOP_Rough

## ----------------------------------------------------------------------------
## test ForwardOP 

data_equ_dir = './classical_method/data'

## domain for solving PDE

g_expre = load_expre(data_equ_dir + '/g_expre.txt')
u_expre = load_expre(data_equ_dir + '/u_expre.txt')
g = fe.Expression(g_expre, degree=2)
u = fe.Expression(u_expre, degree=2)

coordinates = np.load(data_equ_dir + '/coordinates.npy')
freqs = np.load(data_equ_dir + '/freqs.npy')

noise_level = 0.02
data_all_noisy = np.load(data_equ_dir + '/data_all_noisy_' + str(noise_level) + '.npy')
data_all_clean = np.load(data_equ_dir + '/data_all_clean.npy')

tau = 1/(noise_level**2)
noise = Noise(dim=len(coordinates))
noise.set_parameters_Gaussian(covariance_val=1/tau, eval_precision=True)
   
## ----------------------------------------------------------------------------

# forwardOP_rough = ForwardOP_Rough(mesh_size=70, g=g, u=u, f=fe.Constant("1.0"), \
#                            coordinates=coordinates, kappas=freqs, noise=noise)
# # forwardOP_rough.eva_inv()
# forwardOP_rough.load_inv('')
# print("inv matrix loaded")

# forwardOP_rough.to_cuda()
# print("all variables are on gpu")

# a1, a2 = data_all_clean.shape
# data_all = np.zeros((10, 1, a1, a2))
# for ii in range(9):
#     data_all[ii,0,:] = data_all_clean
# data_all[9,0,:] = data_all_noisy
# data_all = torch.tensor(data_all, dtype=torch.float32).cuda()
# start = time.time()
# f_all = forwardOP_rough.inverting_cuda(data_all)
# end = time.time()
# print(end - start)

# f_all = f_all.detach().cpu().numpy()
# f_fun = fe.Function(forwardOP_rough.V)
# f_fun.vector()[:] = f_all[5, -1, :]
# fe.plot(f_fun)

## ----------------------------------------------------------------------------

mesh_size = 200
forwardOP_full = ForwardOP(mesh_size=mesh_size, g=g, u=u, f=fe.Constant("1.0"), \
                            coordinates=coordinates, kappas=freqs)

f_true_expre = load_expre(data_equ_dir + '/f_2D_expre.txt')
f_true = fe.interpolate(fe.Expression(f_true_expre, degree=2), forwardOP_full.domain.function_space)
f_true_vec = f_true.vector()[:]
zz = np.zeros((32,1,len(f_true_vec)))
for ii in range(32):
    zz[ii,0,:] = f_true_vec.copy()
zz = forwardOP_full.vec2matrix(zz)

z = torch.tensor(zz, dtype=torch.float32)
start = time.time()
out = forwardOP_full.forward_op(z)
end = time.time()
print(end - start)

out1 = out[0,0,:]
out2 = out[0,0,:]

data_all_clean = np.load(data_equ_dir + '/data_all_clean.npy')

ii = 29
plt.figure()
plt.plot(data_all_clean[ii,:], label='true')
plt.plot(out1[ii,:], label='current1')
plt.plot(out2[ii,:], label='current2')
plt.legend()