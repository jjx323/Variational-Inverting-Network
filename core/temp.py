#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:31:24 2021

@author: jjx323
"""

import numpy as np
import matplotlib.pyplot as plt


class GaussFourier(object):
    def __init__(self, alpha=1, trunc_num=10, dim=1):
        self.trunc_num, self.dim = trunc_num, dim
        self.alpha = alpha
        self.matrix_phi = None
    
    def set_points(self, x):
        tpi = 2*np.pi
        if x.ndim == 1:
            self.matrix_phi = np.zeros((len(x), np.int(2*self.trunc_num+1)))
            kk = np.linspace(0, self.trunc_num-1, self.trunc_num)
            self.matrix_phi[:, 0:-1:2] = np.array([np.sin(tpi*k*x) for k in kk]).T
            self.matrix_phi[:, 1:-1:2] = np.array([np.cos(tpi*k*x) for k in kk]).T
        elif x.ndim == 2:
            x1d, x2d = x.shape
            self.matrix_phi = np.zeros((x1d, np.int(4*(self.trunc_num**2))))
            kk = np.linspace(0, self.trunc_num-1, self.trunc_num)
            ind = 0
            for k1 in kk:
                for k2 in kk:
                    self.matrix_phi[:, ind] = np.sin(tpi*k1*x[:,0])*np.sin(tpi*k2*x[:,1])
                    ind += 1
                    self.matrix_phi[:, ind] = np.cos(tpi*k1*x[:,0])*np.sin(tpi*k2*x[:,1])
                    ind += 1
                    self.matrix_phi[:, ind] = np.sin(tpi*k1*x[:,0])*np.cos(tpi*k2*x[:,1])
                    ind += 1
                    self.matrix_phi[:, ind] = np.cos(tpi*k1*x[:,0])*np.cos(tpi*k2*x[:,1])
                    ind += 1
            
    def generate_samples(self, sample_num=1):
        if self.dim == 1:
            kk = np.linspace(0, self.trunc_num-1, self.trunc_num) + 1
            xi = np.zeros((np.int(2*self.trunc_num+1), sample_num))
            for i in range(sample_num):
                xi[0:-1:2, i] = np.array([np.random.normal(0, 1/(1e-15+np.power(k, self.alpha))) for k in kk]).ravel()
                xi[1:-1:2, i] = np.array([np.random.normal(0, 1/(1e-15+np.power(k, self.alpha))) for k in kk]).ravel()
            return self.matrix_phi@xi
        if self.dim == 2:
            kk = np.linspace(0, self.trunc_num-1, self.trunc_num) + 1
            xi = np.zeros((np.int(4*(self.trunc_num**2)), sample_num))
            for i in range(sample_num):
                s = 0
                for k1 in kk:
                    for k2 in kk:
                        k1 += 1
                        k2 += 1
                        xi[s, i] = np.random.normal(0, 1/np.power(k1*k1+k2*k2, self.alpha/2))
                        s += 1
            return self.matrix_phi@xi
        
    

if __name__ == '__main__':
#    prior = GaussFourier(1, 1000, 1)
#    x = np.linspace(0, 1, 100)
#    prior.set_points(x)
#    ux = prior.generate_samples(sample_num=2)
#    
#    plt.figure()
#    plt.plot(x, ux[:,0])
#    plt.plot(x, ux[:,1])
    
    prior = GaussFourier(1, 100, 2)
    NN = 100
    x = np.zeros((NN*NN, 2))
    for ind, xi in enumerate(np.linspace(0,1,NN)):
        x[0+NN*ind:100+NN*ind, 0] = xi
        x[0+NN*ind:100+NN*ind, 1] = np.linspace(0,1,NN)
    prior.set_points(x)
    
    ux = prior.generate_samples()
    ux = ux.reshape(NN, NN)
    














