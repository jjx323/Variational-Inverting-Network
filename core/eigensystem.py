#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:38:31 2019

@author: jjx323
"""
import numpy as np
import scipy.linalg as sl
from scipy.sparse.linalg import LinearOperator

#####################################################################
def power_method(Am, dimension=None, k=5, max_iter=500, tol=0.01):
    if dimension is None:
        print("Please specify the dimension")
        return None
    if k > dimension:
        print("k must be smaller than dimension of A")
        return None
    else:
        if type(Am) is np.ndarray:
            def times(x):
                return Am@x
            A = LinearOperator((dimension, dimension), matvec=times)
        else:
            A = Am
        Q = np.random.normal(0, 1, (dimension, k))
        i = 1
        Lam_pre = -10
        diff = 1
        while i <= max_iter and diff > tol: 
            Z = A*Q
            Q, R = sl.qr(Z)
            Lam = np.diag((Q.T)@(A*Q))
            diff = np.max(np.abs(Lam - Lam_pre))
            Lam_pre = Lam
            i = i + 1
        return Lam, Q
            
#####################################################################
def lanczos_method(Am, dimension=None, k=5, max_iter=500, tol=0.01):
    if dimension is None:
        print("Please specify the dimension")
        return None
    if k > dimension:
        print("k must be smaller than dimension of A")
        return None
    else:
        if type(Am) is np.ndarray:
            def times(x):
                return Am@x
            A = LinearOperator((dimension, dimension), matvec=times)
        else:
            A = Am
        Vr = np.zeros((dimension, k))
        T = np.zeros((k, k))
        v = np.random.normal(0, 1, (dimension,))
        v = v/np.sqrt(v@v)
        wp = A*v
        alpha = wp@v
        w = wp - alpha*v
        vp = v.copy()
        Vr[:, 0] = v.copy()
        T[0, 0] = alpha
        j = 2
        while j <= k:
            beta = np.sqrt(w@w)
            if np.abs(beta) > 1e-20:
                v = w/beta
            else:
                v = np.random.normal(0, 1, (dimension,))
                temp = np.zeros((dimension,))
                for i in range(j):
                    temp += (v@Vr[:, i])*Vr[:, i]
                v = v - temp
                v = v/np.sqrt(v@v)
            ## end if
            wp = A*v
            alpha = wp@v
            w = wp - alpha*v - beta*vp
            vp = v.copy()
            if j <= k-1:
                Vr[:, j-1] = v.copy()
                T[j-2, j-1], T[j-1, j-1], T[j-1, j-2] = beta, alpha, beta
            else: 
                T[j-1, j-1] = alpha
            j = j + 1
        ## end while
        val, Q = power_method(T, dimension=k, k=k, max_iter=500)
        return val, Vr@Q, T, Vr
    
    
    
    
    
