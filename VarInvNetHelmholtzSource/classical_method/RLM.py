#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:58 2021

@author: jjx323
"""

import numpy as np
import matplotlib.pyplot as plt

import fenics as fe
import time
import scipy.sparse.linalg as spsl

import shutil
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.misc import load_expre
from core.probability import GaussianMeasure, Noise

from VarInvNetHelmholtzSource.common import EquSolverPML, DomainPML


data_dir = './data'
result_figs_dir = './result/figs'
if os.path.isdir(result_figs_dir):
    shutil.rmtree(result_figs_dir)
os.makedirs(result_figs_dir)

## domain for solving PDE
nn = 120
domain= DomainPML(nx=nn, ny=nn)
domainS = Domain2D(nx=nn, ny=nn, mesh_type='CG', mesh_order=1)

coordinates = np.load(data_dir + '/coordinates.npy')
freqs = np.load(data_dir + '/freqs.npy')
noise_level = 0.05

f_init = fe.Constant("0.0")
f_iter = fe.Function(domain.VR)
f_iter_ = fe.Function(domain.VR)

data_all_noisy = np.load(data_dir + '/data_all_noisy_' + str(noise_level) + '.npy')
data_all_clean = np.load(data_dir + '/data_all_clean.npy')
# data_all_clean = np.load(data_dir + '/d_0.npy')
# data_all_noisy = data_all_clean

# data_all_clean = np.load('./d_0.npy')
# data_all_noisy = data_all_clean
# dl, dr = data_all_clean.shape
# data_all_noise = data_all_clean + noise_level*\
#             np.repeat(np.max(np.abs(data_all_clean), axis=1).reshape(dl,1), dr, axis=1)\
#                     *np.random.randn(dl, dr)

f_true_expre = load_expre(data_dir + '/f_2D_expre.txt')
f_true = fe.interpolate(fe.Expression(f_true_expre, degree=5), domain.VR)
# f_true_data = np.load(data_dir + '/sample_0.npy')
# nn = np.int(np.sqrt(len(f_true_data)))
# domain_true = DomainPML(nx=nn, ny=nn)
# fun_true = fe.Function(domain_true.VR)
# fun_true.vector()[:] = f_true_data
# f_true = fe.interpolate(fun_true, domain.VR)
# f_true = np.load('./u_0.npy')
def relative_error(f, f_true):
    fenzi = fe.assemble(fe.inner(f-f_true, f-f_true)*fe.dx)
    fenmu = fe.assemble(fe.inner(f_true, f_true)*fe.dx)
    return fenzi/fenmu

equ_solver = EquSolverPML(domain=domain, kappa=1, fR=f_init, points=coordinates)
equ_solver.geneForwardNumpyMatrix()
equ_solver.geneAdjointNumpyMatrix()

start = time.time()
for ii, freq in enumerate(freqs):
    equ_solver.update_k(freq)
    equ_solver.update_f(f_iter.vector()[:])
    equ_solver.forward_solve()
    data = equ_solver.get_data(equ_solver.forward_sol_vec)
    equ_solver.adjoint_solve(data - data_all_noisy[ii, :])
    grad_val = equ_solver.adjoint_sol_vec[::2] 
    gnorm = np.max(np.abs(grad_val)) + 1e-15
    
    step_length = 0.05
        
    f_iter.vector()[:] += -step_length*grad_val/gnorm
    print("freq = {:.2f}, relative_error = {:.2f}, step_length = {:.5f}".format(freq, \
                                                  relative_error(f_iter, f_true), step_length))
    
    plt.figure(figsize=(13,5))
    plt.subplot(1,2,1)
    fig = fe.plot(f_iter)
    plt.colorbar(fig)
    plt.title("Estimated f")
    plt.subplot(1,2,2)
    fig = fe.plot(f_true)
    # fig = plt.imshow(f_true)
    plt.colorbar(fig)
    plt.title("True f")
    plt.savefig(result_figs_dir + '/fig_' + str(ii) + '.png')
    plt.close()

end = time.time()
print("Total Time: ", end-start)


