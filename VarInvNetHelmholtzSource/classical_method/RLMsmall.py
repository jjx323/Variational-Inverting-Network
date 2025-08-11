#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:14:23 2021

@author: jjx323
"""

import numpy as np
import matplotlib.pyplot as plt

import fenics as fe
import time

import shutil
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.misc import load_expre
from core.probability import GaussianMeasure, Noise

from VarInvNetHelmholtzSource.common import DomainPML, EquSolverPML, EquSolverPMLSmall


data_dir = './data'
result_figs_dir = './result/figs_50'
if os.path.isdir(result_figs_dir):
    shutil.rmtree(result_figs_dir)
os.makedirs(result_figs_dir)

## domain for solving PDE
nn = 30
domainPML= DomainPML(nx=nn, ny=nn)
domainS = Domain2D(nx=nn, ny=nn, mesh_type='CG', mesh_order=1)

coordinates = np.load(data_dir + '/coordinates.npy')
freqs = np.load(data_dir + '/freqs.npy')
noise_level = 0.05

f_init = fe.interpolate(fe.Constant("0.0"), domainPML.VR)
f_iter = fe.interpolate(f_init, domainPML.VR)
f_iter_ = fe.interpolate(f_init, domainPML.VR)

data_all_noisy = np.load(data_dir + '/data_all_noisy_' + str(noise_level) + '.npy')
data_all_clean = np.load(data_dir + '/data_all_clean.npy')
# data_all_clean = np.load(data_dir + '/dr_0.npy')

# data_all_clean = np.load('./d_0.npy')
# data_all_noisy = data_all_clean

equ_solvers = []
for ii, freq in enumerate(freqs):
    equ_solvers.append(EquSolverPMLSmall(domain=domainPML, kappa=freq, fR=f_init, points=coordinates))
    Ainv, Binv = equ_solvers[ii].eva_inv()
    np.save(data_dir + '/invForwardMatrix_' + str(ii), Ainv)
    np.save(data_dir + '/invAdjointMatrix_' + str(ii), Binv)
    print("inv ii = {:2}, freq = {:.2f}".format(ii, freq))
print("Preparations are completed!")
equ_solver = EquSolverPMLSmall(domain=domainPML, kappa=1.0, fR=f_init, points=coordinates)
equ_solver.to_dense()

# for ii, freq in enumerate(freqs):
#     equ_solver.update_k(freq)
#     Ainv, Binv = equ_solver.eva_inv()
#     np.save(data_dir + '/invForwardMatrix_' + str(ii), Ainv)
#     np.save(data_dir + '/invAdjointMatrix_' + str(ii), Binv) 
    
Ainvs, Binvs = [], []
for ii, freq in enumerate(freqs):
    Ainv = np.array(np.load(data_dir + '/invForwardMatrix_' + str(ii) + '.npy'))
    Binv = np.array(np.load(data_dir + '/invAdjointMatrix_' + str(ii) + '.npy'))
    Ainvs.append(Ainv)
    Binvs.append(Binv)

f_true_expre = load_expre(data_dir + '/f_2D_expre.txt')
f_true = fe.interpolate(fe.Expression(f_true_expre, degree=5), domainPML.VR)
# f_true = np.load('./u_0.npy')
def relative_error(f, f_true):
    fenzi = fe.assemble(fe.inner(f-f_true, f-f_true)*fe.dx)
    fenmu = fe.assemble(fe.inner(f_true, f_true)*fe.dx)
    return fenzi/fenmu

start = time.time()
for ii, freq in enumerate(freqs):
    for mm in range(1):
        Ainv = Ainvs[ii] 
        Binv = Binvs[ii] 
        equ_solver.update_k(freq)
        equ_solver.update_f(f_iter.vector()[:])
        sol_forward = equ_solver.forward_solve(Ainv)
        # sol_forward = equ_solver.forward_solve()
        res_vec = equ_solver.get_data(sol_forward) - data_all_noisy[ii, :]
        sol_adjoint = equ_solver.adjoint_solve(res_vec, Binv)
        # sol_adjoint = equ_solver.adjoint_solve(res_vec)
        grad_val = sol_adjoint[::2] 
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
print(end-start)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # plt.figure(figsize=(13,5))
    # plt.subplot(1,2,1)
    # fig = fe.plot(f_iter)
    # plt.colorbar(fig)
    # plt.title("Estimated f")
    # plt.subplot(1,2,2)
    # fig = fe.plot(f_true)
    # plt.colorbar(fig)
    # plt.title("True f")
    # plt.savefig(result_figs_dir + '/fig_' + str(ii) + '.png')
    # plt.close()