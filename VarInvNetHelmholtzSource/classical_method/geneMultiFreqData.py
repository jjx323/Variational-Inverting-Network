#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:17:38 2021

@author: jjx323
"""

import numpy as np
# import matplotlib.pyplot as plt

import fenics as fe

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.misc import save_expre

from VarInvNetHelmholtzSource.common import EquSolverPML, DomainPML


data_dir = './data'
if os.path.isdir(data_dir) == False:
    os.makedirs(data_dir)

## domain for solving PDE
nn = 300
domain= DomainPML(nx=nn, ny=nn)

f_expre01 = '0.3*pow(1-3*(x[0]*2-1), 2)*exp(-pow(3*(x[0]*2-1), 2)-pow(3*(x[1]*2-1)+1, 2))'
f_expre02 = '- (0.2*3*(x[0]*2-1) - pow(3*(x[0]*2-1), 3)-pow(3*(x[1]*2-1), 5))'
f_expre03 = '*exp(-pow(3*(x[0]*2-1),2)-pow(3*(x[1]*2-1),2))'
f_expre04 = '- 0.03*exp(-pow(3*(x[0]*2-1)+1, 2)-pow(3*(x[1]*2-1),2))'
f_expre0 = f_expre01 + f_expre02 + f_expre03 + f_expre04
f_expre1 = 'pow((2*x[0]-1), 2) + pow((2*x[1]-1), 2) < 0.7 ?' + f_expre0 + ' : '
f_expre2 = 'pow((2*x[0]-1), 2) + pow((2*x[1]-1), 2) > 0.9 ? 0.0 : -0.5'
f_expre = f_expre1 + f_expre2
# f_expre = f_expre0
f = fe.interpolate(fe.Expression(f_expre, degree=3), domain.V1)

file2 = fe.File('./saved_mesh.xml')
file2 << domain.mesh

save_expre(data_dir + '/f_2D_expre.txt', f_expre)

## specify the measurement points
coordinates = []
Nx, Ny = 20, 20
# Nx, Ny = 50, 50
for j in range(Ny):
    coordinates.append([0.0, j/Ny])
for j in range(Ny):
    coordinates.append([1.0, j/Ny])
for j in range(Ny):
    coordinates.append([j/Ny, 0.0])
for j in range(Ny):
    coordinates.append([j/Ny, 1.0])
coordinates = np.array(coordinates)
data_length = len(coordinates)
np.save(data_dir + '/coordinates', coordinates)

num_freqs = 48
# num_freqs = 30
freqs = np.linspace(1, 50, num_freqs)
np.save(data_dir + '/freqs', freqs)

equ_solver = EquSolverPML(domain=domain, kappa=5, fR=f, points=coordinates)
equ_solver.geneForwardNumpyMatrix()

data_all_clean = np.zeros((num_freqs, 4*(Nx+Ny)))
data_all_noisy = np.zeros((num_freqs, 4*(Nx+Ny)))

noise_level = 0.05
for ii, freq in enumerate(freqs):
    equ_solver.update_k(freq)
    equ_solver.forward_solve()
    data_all_clean[ii, :] = np.array(equ_solver.get_data(equ_solver.forward_sol_vec))
    a = max(data_all_clean[ii, :])
    data_all_noisy[ii, :] = data_all_clean[ii, :] \
                    + noise_level*a*np.random.normal(0, 1, (len(data_all_clean[ii,:]),))
    print("ii = {:2}".format(ii))

np.save(data_dir + '/data_all_clean', data_all_clean)
np.save(data_dir + '/data_all_noisy_' + str(noise_level), data_all_noisy)






