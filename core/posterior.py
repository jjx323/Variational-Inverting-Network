#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:13:03 2020

@author: jjx323
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


class PosteriorLaplaceApprox(object):
    def __init__(self, model, mean_vec=None):
        self.model = model
        self.dim = self.model.domain_coef.function_space.dim()
        if type(mean_vec) != np.ndarray:
            mean_vec = np.zeros(self.dim)
        self.mean_vec = mean_vec
        self.M = self.model.M_coef
        self.Mdiag = sps.diags(self.model.M_coef.diagonal())
        self.Mfhalf = sps.diags(np.power(self.Mdiag.diagonal(), -0.5))
        self.VDV, self.VLamV, self.VPV, self.eigLam, self.eigV = None, None, None, None, None
        self.has_Hessian = False
        self.Id = sps.diags(np.ones(self.dim))
        self.K = self.model.ref_u.K
        self.L = None
        self.MA2 = self.K@spsl.spsolve(self.M, self.K)
        
    def H_tilde_misfit(self, x):
        val = self.model.M_coef@self.model.eval_hessian_res_vec(x)
        return val    

    def H_tilde_misfit_operator(self):
        leng = self.model.M_coef.shape[0]
        linear_ope = spsl.LinearOperator((leng, leng), matvec=self.H_tilde_misfit)
        return linear_ope
        
    def decompose_Hessian(self, k=50, maxiter=None, v0=None):
        operator = self.H_tilde_misfit_operator()
        if type(v0) is np.ndarray:
            v0 = v0
        if maxiter != None:
            maxiter = maxiter
            
        self.eigLam, self.eigV = spsl.eigsh(operator, M=self.MA2, k=k, which='LM', maxiter=maxiter, v0=v0) 

        index = self.eigLam > 0.7
        if np.sum(index) == k:
            print("Warring! The eigensystem may be inaccurate!")
        self.eigLam = np.flip(self.eigLam[index])
        self.eigV = np.flip(self.eigV[:, index], axis=1)
        self.eigV = spsl.spsolve(self.model.M_coef, self.model.ref_u.K@self.eigV)
        #eigVd = eigV.T@model_pre.M_coef
        eigLamD = self.eigLam/(self.eigLam + 1)
        #VDVd = eigV@sps.diags(eigLamD)@eigVd
        # print(self.eigV.shape, sps.diags(eigLamD).shape)
        self.VDV = self.eigV@sps.diags(eigLamD)@self.eigV.T
        self.VLamV = self.eigV@sps.diags(self.eigLam)@self.eigV.T
        #val, VrQ, T, Vr = lanczos_method(operator, model_pre.M_coef.shape[0], k=20)
        eigLamP = 1/np.sqrt(self.eigLam+1)-1
        self.VPV = self.eigV@sps.diags(eigLamP)@self.eigV.T
        self.has_Hessian = True
        self.L = self.M@(self.VPV@self.M + self.Id)@self.Mfhalf

    def generate_sample(self):
        if self.has_Hessian == True:
            x = np.random.normal(0, 1, (self.dim,))
            val = spsl.spsolve(self.K, (self.L@x).T) + self.mean_vec
            return val
        else:
            print("Please run decompose_Hessian first !")
            return None
        











