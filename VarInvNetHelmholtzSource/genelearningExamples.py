#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:32:03 2021

@author: jjx323
"""

import numpy as np
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt

import fenics as fe
import time
import shutil

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.misc import save_expre

from VarInvNetHelmholtzSource.common import EquSolverPML, DomainPML


data_dir = './data_train_test/classical_method_data'
mesh_dir = './data_train_test'
samples_dir = './data_train_test/samples_all'

## domain for solving PDE
nn = 500
domainPML = DomainPML(nx=nn, ny=nn)
domainS = Domain2D(nx=nn, ny=nn, mesh_type='CG', mesh_order=1)

fun = fe.Function(domainPML.VR)
np.save(mesh_dir + "/fun_vec", fun.vector()[:])
file1 = fe.File(mesh_dir + "/fun.xml")
file1 << fun
file2 = fe.File(mesh_dir + '/saved_mesh_' + str(nn) + '.xml')
file2 << domainPML.VR.mesh()

## specify the measurement points
coordinates = []
Nx, Ny = 20, 20
# Nx, Ny = 30, 30
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
freqs = np.linspace(1, 50, num_freqs)
np.save(data_dir + '/freqs', freqs)

equ_solver = EquSolverPML(domain=domainPML, kappa=1, points=coordinates)
equ_solver.geneForwardNumpyMatrix()
equ_solver.geneAdjointNumpyMatrix()

def get_data(sol, equ_solver):
    SR, SI = equ_solver.SR, equ_solver.SI
    dataR = SR@sol[::2,:]
    dataI = SI@sol[1::2,:]
    h, l = dataR.shape 
    data = np.zeros((2*h, l))
    for i in range(h):
        data[2*i, :] = dataR[i, :]
        data[2*i+1, :] = dataI[i, :]
    return data

# ## ----------------------------------------------------------------------------

# f_expre01 = "pow(1-pow((2*x[0]-1),2), a11)*pow(1-pow((2*x[1]-1),2),a12)*a13*" + \
#             "exp(-a14*pow((2*x[0]-1)-a15,2)-a16*pow((2*x[1]-1)-a17,2))"
# f_expre02 = "pow(1-pow((2*x[0]-1),2), a21)*pow(1-pow((2*x[1]-1),2),a22)*a23*" + \
#             "exp(-a24*pow((2*x[0]-1)-a25,2)-a26*pow((2*x[1]-1)-a27,2))"
# f_expre03 = "pow(1-pow((2*x[0]-1),2), a31)*pow(1-pow((2*x[1]-1),2),a32)*a33*" + \
#             "exp(-a34*pow((2*x[0]-1)-a35,2)-a36*pow((2*x[1]-1)-a37,2))"
# f_expre0 = f_expre01 + '+' + f_expre02 + '+' + f_expre03
# f_expre1 = 'pow((2*x[0]-1), 2) + pow((2*x[1]-1), 2) < r1 ?' + f_expre0 + ' : '
# f_expre2 = 'pow((2*x[0]-1), 2) + pow((2*x[1]-1), 2) > r2 ? 0.0 : -0.5'
# f_expre = f_expre1 + f_expre2
# # f_expre = f_expre0

# num_samples = 1500
# f_all = []
# for ii in range(0, num_samples):
#     start = time.time()
#     a11, a21, a31 = np.random.uniform(1,3), np.random.uniform(1,3), np.random.uniform(1,3)
#     a12, a22, a32 = np.random.uniform(1,3), np.random.uniform(1,3), np.random.uniform(1,3)
#     a13, a23, a33 = np.random.uniform(-1,1), np.random.uniform(-1,1), \
#                     np.random.uniform(-1,1)
#     a14, a24, a34 = np.random.uniform(8,10), np.random.uniform(8,10), np.random.uniform(8,10)
#     a16, a26, a36 = np.random.uniform(8,10), np.random.uniform(8,10), np.random.uniform(8,10)
#     a15, a25, a35 = np.random.uniform(-0.7, 0.7), np.random.uniform(-0.7, 0.7), \
#                     np.random.uniform(-0.7, 0.7)
#     a17, a27, a37 = np.random.uniform(-0.7, 0.7), np.random.uniform(-0.7, 0.7), \
#                     np.random.uniform(-0.7, 0.7)
#     r1 = np.random.uniform(0.65, 0.75)
#     r2 = np.random.uniform(0.85, 0.95)
#     f = fe.interpolate(fe.Expression(f_expre, degree=5, \
#                                     a11=a11, a21=a21, a31=a31, \
#                                     a12=a12, a22=a22, a32=a32, \
#                                     a13=a13, a23=a23, a33=a33, \
#                                     a14=a14, a24=a24, a34=a34, \
#                                     a16=a16, a26=a26, a36=a36, \
#                                     a15=a15, a25=a25, a35=a35, \
#                                     a17=a17, a27=a27, a37=a37, \
#                                     r1=r1, r2=r2), domainPML.VR)
#     f_all.append(f.vector()[:])
    
# f_all = np.array(f_all)
# np.save(samples_dir + '/samples_f_', f_all)

# f_all = np.array(np.load(samples_dir + '/samples_f_.npy'))
# ## -------------------------------------------------------
# ## save samples for training and testing 

# mulu = './data_train_test/samples'
# if os.path.isdir(mulu):
#     shutil.rmtree(mulu)
# os.makedirs(mulu)
# mulu = './data_train_test/test_data'
# if os.path.isdir(mulu):
#     shutil.rmtree(mulu)
# os.makedirs(mulu)

# for ii in range(1000):
#     np.save('./data_train_test/samples/sample_' + str(ii), np.array(f_all[ii,:]))

# for ii in range(500):
#     np.save('./data_train_test/test_data/sample_' + str(ii), np.array(f_all[ii+1000,:]))

# ## -------------------------------------------------------

# num_parallel = 500
# data_all_clean = np.zeros((num_freqs, 4*(Nx+Ny), num_parallel))

# for ii in range(0, np.int(num_samples/num_parallel)):
#     for jj, freq in enumerate(freqs):
#         equ_solver.update_k(freq)
#         A = equ_solver.A1ForwardNumpy + freq*freq*equ_solver.A2ForwardNumpy
#         temp1 = f_all[ii*num_parallel:(ii+1)*num_parallel, :].T
#         h, l = temp1.shape
#         b1 = np.zeros((2*h, l))
#         for hangshu in range(h):
#             b1[2*hangshu, :] = temp1[hangshu, :]
#         b = equ_solver.M@b1
#         sol = spsl.spsolve(A, b)
#         data_all_clean[jj, :, :] = np.array(get_data(sol, equ_solver))
#         print("num_samples = {:4}, num_freq = {:2}".format(ii, jj))
    
#     np.save(samples_dir + '/sample_' + str(ii), data_all_clean)
#     print("ii = ", ii)

# ## -------------------------------------------------------
# ## save data for training and testing

# learn1 = np.load(samples_dir + '/sample_0.npy')
# learn2 = np.load(samples_dir + '/sample_1.npy')
# test = np.load(samples_dir + '/sample_2.npy')

# mulu = './data_train_test/d_examples_learn'
# if os.path.isdir(mulu):
#     shutil.rmtree(mulu)
# os.makedirs(mulu)

# _, _, n1 = learn1.shape 
# for ii in range(n1):
#     np.save(mulu + '/d_' + str(ii), learn1[:,:,ii])
# _, _, n2 = learn2.shape 
# for ii in range(n2):
#     np.save(mulu + '/d_' + str(ii+n1), learn2[:,:,ii])

# mulu = './data_train_test/d_examples_test'
# if os.path.isdir(mulu):
#     shutil.rmtree(mulu)
# os.makedirs(mulu)

# _, _, nn = test.shape 
# for ii in range(nn):
#     np.save(mulu + '/d_' + str(ii), test[:,:,ii])

## ----------------------------------------------------------
## generate test examples combine two functions

# f_expre01 = "pow(1-pow((2*x[0]-1),2), a11)*pow(1-pow((2*x[1]-1),2),a12)*a13*" + \
#             "exp(-a14*pow((2*x[0]-1)-a15,2)-a16*pow((2*x[1]-1)-a17,2))"
# f_expre02 = "pow(1-pow((2*x[0]-1),2), a21)*pow(1-pow((2*x[1]-1),2),a22)*a23*" + \
#             "exp(-a24*pow((2*x[0]-1)-a25,2)-a26*pow((2*x[1]-1)-a27,2))"
# f_expre03 = "pow(1-pow((2*x[0]-1),2), a31)*pow(1-pow((2*x[1]-1),2),a32)*a33*" + \
#             "exp(-a34*pow((2*x[0]-1)-a35,2)-a36*pow((2*x[1]-1)-a37,2))"
# f_expre0 = f_expre01 + '+' + f_expre02 + '+' + f_expre03
# f_expre1 = 'pow((2*x[0]-1), 2) + pow((2*x[1]-1), 2) < r1 ?' + f_expre0 + ' : '
# f_expre2 = 'pow((2*x[0]-1), 2) + pow((2*x[1]-1), 2) > r2 ? 0.0 : -0.5'
# f_expre = f_expre1 + f_expre2
# # f_expre = f_expre0

# f2_expre01 = "pow(1-pow((2*x[0]-1),2), a11)*pow(1-pow((2*x[1]-1),2),a12)*a13*" + \
#             "exp(-a14*pow((2*x[0]-1)-a15,2)-a16*pow((2*x[1]-1)-a17,2))"
# f2_expre02 = "pow(1-pow((2*x[0]-1),2), a21)*pow(1-pow((2*x[1]-1),2),a22)*a23*" + \
#             "exp(-a24*pow((2*x[0]-1)-a25,2)-a26*pow((2*x[1]-1)-a27,2))"
# f2_expre03 = "pow(1-pow((2*x[0]-1),2), a31)*pow(1-pow((2*x[1]-1),2),a32)*a33*" + \
#             "exp(-a34*pow((2*x[0]-1)-a35,2)-a36*pow((2*x[1]-1)-a37,2))"
# f2_expre0 = f2_expre01 + '+' + f2_expre02 + '+' + f2_expre03
# f2_expre1 = 'pow((2*x[0]-1), 2) + pow((2*x[1]-1), 2) < 0.95 ?' + f2_expre0 + ' : '
# f2_expre2 = 'pow((2*x[0]-1), 2) + pow((2*x[1]-1), 2) > 0.95 ? 0.0 : 0.0'
# f2_expre = f2_expre1 + f2_expre2

# num_samples = 500
# f_all = []
# for ii in range(0, num_samples):
#     a11, a21, a31 = np.random.uniform(1,3), np.random.uniform(1,3), np.random.uniform(1,3)
#     a12, a22, a32 = np.random.uniform(1,3), np.random.uniform(1,3), np.random.uniform(1,3)
#     a13, a23, a33 = np.random.uniform(-1,1), np.random.uniform(-1,1), \
#                     np.random.uniform(-1,1)
#     a14, a24, a34 = np.random.uniform(8,10), np.random.uniform(8,10), np.random.uniform(8,10)
#     a16, a26, a36 = np.random.uniform(8,10), np.random.uniform(8,10), np.random.uniform(8,10)
#     a15, a25, a35 = np.random.uniform(-0.7, 0.7), np.random.uniform(-0.7, 0.7), \
#                     np.random.uniform(-0.7, 0.7)
#     a17, a27, a37 = np.random.uniform(-0.7, 0.7), np.random.uniform(-0.7, 0.7), \
#                     np.random.uniform(-0.7, 0.7)
#     r1 = np.random.uniform(0.65, 0.75)
#     r2 = np.random.uniform(0.85, 0.95)
#     f1 = fe.interpolate(fe.Expression(f_expre, degree=5, \
#                                     a11=a11, a21=a21, a31=a31, \
#                                     a12=a12, a22=a22, a32=a32, \
#                                     a13=a13, a23=a23, a33=a33, \
#                                     a14=a14, a24=a24, a34=a34, \
#                                     a16=a16, a26=a26, a36=a36, \
#                                     a15=a15, a25=a25, a35=a35, \
#                                     a17=a17, a27=a27, a37=a37, \
#                                     r1=r1, r2=r2), domainPML.VR)

#     a11, a21, a31 = np.random.uniform(1,3), np.random.uniform(1,3), np.random.uniform(1,3)
#     a12, a22, a32 = np.random.uniform(1,3), np.random.uniform(1,3), np.random.uniform(1,3)
#     a13, a23, a33 = np.random.uniform(-1,1), np.random.uniform(-1,1), \
#                     np.random.uniform(-1,1)
#     a14, a24, a34 = np.random.uniform(8,10), np.random.uniform(8,10), np.random.uniform(8,10)
#     a16, a26, a36 = np.random.uniform(8,10), np.random.uniform(8,10), np.random.uniform(8,10)
#     a15, a25, a35 = np.random.uniform(-0.7, 0.7), np.random.uniform(-0.7, 0.7), \
#                     np.random.uniform(-0.7, 0.7)
#     a17, a27, a37 = np.random.uniform(-0.7, 0.7), np.random.uniform(-0.7, 0.7), \
#                     np.random.uniform(-0.7, 0.7)

#     f2 = fe.interpolate(fe.Expression(f2_expre, degree=5, \
#                                     a11=a11, a21=a21, a31=a31, \
#                                     a12=a12, a22=a22, a32=a32, \
#                                     a13=a13, a23=a23, a33=a33, \
#                                     a14=a14, a24=a24, a34=a34, \
#                                     a16=a16, a26=a26, a36=a36, \
#                                     a15=a15, a25=a25, a35=a35, \
#                                     a17=a17, a27=a27, a37=a37), domainPML.VR)
    
#     # f = fe.Function(domainPML.VR)
#     # f.vector()[:] = f1.vector()[:] + f2.vector()[:]
#     # f = fe.interpolate(f1+f2, domainPML.VR)
#     f_all.append(f1.vector()[:] + f2.vector()[:])
    
# f_all = np.array(f_all)
# np.save(samples_dir + '/samples_f2_', f_all)

# f_all = np.array(np.load(samples_dir + '/samples_f2_.npy'))


# num_parallel = 500
# data_all_clean = np.zeros((num_freqs, 4*(Nx+Ny), num_parallel))

# for ii in range(0, np.int(num_samples/num_parallel)):
#     for jj, freq in enumerate(freqs):
#         equ_solver.update_k(freq)
#         A = equ_solver.A1ForwardNumpy + freq*freq*equ_solver.A2ForwardNumpy
#         temp1 = f_all[ii*num_parallel:(ii+1)*num_parallel, :].T
#         h, l = temp1.shape
#         b1 = np.zeros((2*h, l))
#         for hangshu in range(h):
#             b1[2*hangshu, :] = temp1[hangshu, :]
#         b = equ_solver.M@b1
#         sol = spsl.spsolve(A, b)
#         data_all_clean[jj, :, :] = np.array(get_data(sol, equ_solver))
#         print("num_samples = {:4}, num_freq = {:2}".format(ii, jj))
    
#     np.save(samples_dir + '/sample2_' + str(ii), data_all_clean)
#     print("ii = ", ii)

## ----------------------------------------------------------------------------
## generate data with different freqs

samples_f = np.load('./data_train_test/samples_all/samples_f_.npy')[1000:,:]
samples_f2 = np.load('./data_train_test/samples_all/samples_f2_.npy')

num_freqs = 60
freqs = np.linspace(1, 60, num_freqs)

data_all_clean = np.zeros((num_freqs, 4*(Nx+Ny), 500))

for jj, freq in enumerate(freqs):
    equ_solver.update_k(freq)
    A = equ_solver.A1ForwardNumpy + freq*freq*equ_solver.A2ForwardNumpy
    temp1 = samples_f.T
    h, l = temp1.shape
    b1 = np.zeros((2*h, l))
    for hangshu in range(h):
        b1[2*hangshu, :] = temp1[hangshu, :]
    b = equ_solver.M@b1
    sol = spsl.spsolve(A, b)
    data_all_clean[jj, :, :] = np.array(get_data(sol, equ_solver))
    print("num_freq = {:2}".format(jj))

np.save('./data_train_test/samples_all/' + '/sample_60fre', data_all_clean)

## ----------------------------------------------------------------------------

data_all_clean = np.zeros((num_freqs, 4*(Nx+Ny), 500))

for jj, freq in enumerate(freqs):
    equ_solver.update_k(freq)
    A = equ_solver.A1ForwardNumpy + freq*freq*equ_solver.A2ForwardNumpy
    temp1 = samples_f2.T
    h, l = temp1.shape
    b1 = np.zeros((2*h, l))
    for hangshu in range(h):
        b1[2*hangshu, :] = temp1[hangshu, :]
    b = equ_solver.M@b1
    sol = spsl.spsolve(A, b)
    data_all_clean[jj, :, :] = np.array(get_data(sol, equ_solver))
    print("num_freq = {:2}".format(jj))

np.save('./data_train_test/samples_all/' + '/sample2_60fre', data_all_clean)















