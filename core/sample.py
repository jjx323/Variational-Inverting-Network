#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:59:05 2019

@author: jjx323
"""
import numpy as np
import os

###################################################################
class pCN(object):
    '''
    M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems, 
    Hankbook of Uncertainty Quantification, 2017
    '''
    def __init__(self, prior, phi, beta=0.01, save_num=np.int(1e4), path=None):
        self.prior = prior
        self.phi = phi
        self.beta = beta
        self.dt = (2*(2 - beta**2 - 2*np.sqrt(1-beta**2)))/(beta**2)
        self.save_num = save_num
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        
    def generate_chain(self, length_total=1e5, callback=None, uk=None, index=None):
        chain = []
        if uk is None:
            uk = self.prior.generate_sample(flag='only_vec')
        else:
            uk = uk
        chain.append(uk.copy())
        ac_rate = 0
        ac_num = 0
        
        def aa(u_new, phi_old):
            #return min(1, np.exp(self.phi(u_old)-self.phi(u_new)))
            #print(self.phi(u_old)-self.phi(u_new))
            phi_new = self.phi(u_new)
            panduan = np.exp(min(0.0, phi_old-phi_new))
            return panduan, phi_new
        
        si = 0
        if index == None: index = 0
        phi_old = 1e20   # a large enough number 
        i = 1
        while i <= length_total:
            a = np.sqrt(1-self.beta*self.beta)
            b = self.beta
            xik = self.prior.generate_sample(flag='only_vec')
            vk = a*uk + b*xik
            t, phi_new = aa(vk, phi_old)
            r = np.random.uniform(0, 1)
            if t >= r:
                chain.append(vk.copy())
                uk = vk.copy()
                ac_num += 1
                phi_old = phi_new
            else: 
                chain.append(uk.copy())
            ac_rate = ac_num/i 
            i += 1
            
            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_' + np.str(np.int(index)), chain)
                    del chain
                    chain = []
                    index += 1
                    
#            print(i, ' , ', round(ac_rate, 4), ' , ', self.beta)
        
            if callback is not None:
                callback([uk, i, ac_rate])

        if self.path is None:
            return [chain, ac_rate]
        else:
            return [ac_rate, self.path, np.int(index)]

###################################################################
class pCN_FMALA(pCN):
    '''
    [1] M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems,
    Hankbook of Uncertainty Quantification, 2017
    [2] T. Bui-Thanh, Q. P. Nguyen, FEM-based discretization-invariant MCMC methods
    for PDE-constrained Bayesian inverse problems, Inverse Problems and Imaging,
    10(4), 2016, 943-975
    '''
    def __init__(self, model, grad_eval, phi, beta=0.01, save_num=np.int(1e4), path=None):
        super().__init__(model.ref_u, phi, beta, save_num, path)
        self.model = model
        self.a, self.b = np.sqrt(1-self.beta*self.beta), self.beta
        self.c = -2*self.dt/(2+self.dt)
        self.grad_eval = grad_eval

    def aa(self, u_new, u_old, g_old):
        temp1 = (self.prior.c_half@g_old)
        temp1 = (temp1)@self.model.M_coef@(temp1.T)
        rho_old_new = self.phi(u_old) + 0.5*g_old@self.model.M_coef@(u_new-u_old) \
                  + self.dt/4*g_old@self.model.M_coef@(u_old+u_new) + self.dt/4*temp1

        g_new = self.grad_eval(self.model, u_new)
        temp2 = self.prior.c_half@g_new
        temp2 = (temp2)@self.model.M_coef@(temp2.T)
        rho_new_old = self.phi(u_new) + 0.5*g_new@self.model.M_coef@(u_old-u_new) \
                  + self.dt/4*g_new@self.model.M_coef@(u_old+u_new) + self.dt/4*temp2

        # print(temp1, temp2, np.exp(min(0.0, rho_old_new - rho_new_old)))
        return np.exp(min(0.0, rho_old_new - rho_new_old))

    def generate_chain(self, length_total=1e5, callback=None, uk=None):
        chain = []
        if uk is None:
            uk = self.prior.generate_sample(flag='only_vec')
        else:
            uk = uk
        chain.append(uk.copy())
        ac_rate = 0
        ac_num = 0

        i=1
        si, index = 0, 0
        while i <= length_total:
            xik = self.prior.generate_sample(flag='only_vec')
            g_old = self.grad_eval(self.model, uk)
            vk = self.a*uk + self.c*self.prior.c@g_old + self.b*xik
            t = self.aa(vk, uk, g_old)
            r = np.random.uniform(0, 1)
            if t >= r:
                chain.append(vk.copy())
                uk = vk.copy()
                ac_num += 1
            else:
                chain.append(uk.copy())
            ac_rate = ac_num/i
            i += 1

            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_' + np.str(np.int(index)), chain)
                    del chain
                    chain = []
                    index += 1

#            print(i, ' , ', round(ac_rate, 4), ' , ', self.beta)

            if callback is not None:
                callback([uk, i, ac_rate])

        if self.path is None:
            return [chain, ac_rate]
        else:
            return [ac_rate, self.path, np.int(index)]

###################################################################
class SpCN(object):
    '''
    Z. Yao, Z. Hu and J. Li, A TV-Gaussian prior for infinite-dimensional
    Bayesian inverse problems and its numerical implementations,
    Inverse Problems, 32, 2016, 075006
    '''
    def __init__(self, prior, R, phi, beta=0.01, save_num=np.int(1e4), path=None):
        self.prior = prior
        self.R = R
        self.phi = phi
        self.beta = beta
        self.dt = (2*(2 - beta**2 - 2*np.sqrt(1 - beta**2)))/(beta**2)
        self.save_num = save_num
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    # def accept_R(self, v_new, v_old):
    #     #return min(1, np.exp(self.R(v_old) - self.R(v_new)))
    #     return np.exp(min(0.0, self.R(v_old) - self.R(v_new)))

    def s_sample(self, v_old, K):
        for i in range(K):
            w = self.prior.generate_sample(flag='only_vec')
            a, b = np.sqrt(1-self.beta*self.beta), self.beta
            v_new = a*v_old + b*w
            # acc = self.accept_R(v_new, v_old)
            acc = np.exp(min(0.0, self.R(v_old) - self.R(v_new)))
            n = np.random.uniform(0, 1)
            if acc >= n:
                v_old = v_new.copy()
                # print(i)
                break
        return v_old

    def generate_chain(self, length_total=1e5, K=20, callback=None):
        chain = []
        ac_rate = 0
        ac_num = 0
        _, u_old = self.prior.generate_sample()
        chain.append(u_old.copy())

        def aa(u_new, u_old):
            #return min(1, np.exp(self.phi(u_old)-self.phi(u_new)))
            return np.exp(min(0.0, self.phi(u_old)-self.phi(u_new)))

        i = 1
        si, index = 0, 0
        while i <= length_total:
            u_new = self.s_sample(u_old, K=K)
            acc = aa(u_new, u_old)
            n = np.random.uniform(0, 1)
            if acc >= n:
                chain.append(u_new.copy())
                u_old = u_new.copy()
                ac_num += 1
            else:
                chain.append(u_old.copy())
            i += 1

            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_' + np.str(np.int(index)), chain)
                    del chain
                    chain = []
                    index += 1

            ac_rate = ac_num/i

            if callback is not None:
                callback([u_old, i, ac_rate])

        if self.path is None:
            return [chain, ac_rate]
        else:
            return [ac_rate, self.path, np.int(index)]

###################################################################
class SpCN_FMALA(SpCN):
    '''
    [1] Z. Yao, Z. Hu and J. Li, A TV-Gaussian prior for infinite-dimensional
    Bayesian inverse problems and its numerical implementations,
    Inverse Problems, 32, 2016, 075006
    [2] T. Bui-Thanh, Q. P. Nguyen, FEM-based discretization-invariant MCMC methods
    for PDE-constrained Bayesian inverse problems, Inverse Problems and Imaging,
    10(4), 2016, 943-975

    In this SpCN_FMALA version, we used Function-space Metropolis-adjusted Langevin algorithm (FMALA)
    in generating samples from the hybrid prior.
    The term \lam*R(u) are seem as the residual term in [2].
    '''
    def __init__(self, model, phi, beta=0.01, res_FMALA=True, save_num=np.int(1e4), path=None):
        fun_R = lambda x: model.lam*self.model.R.evaluate_R(x)
        super().__init__(model.ref_u, fun_R, phi, beta, save_num, path)
        self.eva_grad_R = lambda x: model.lam*model.R.evaluate_grad_R(x)
        self.model = model
        self.res_FMALA = res_FMALA

    def accept_R(self, v_new, v_old, g_old):
        temp1 = (self.prior.c_half@g_old)
        temp1 = np.sum(temp1*(self.model.M_coef@temp1))
        rho_old_new = self.R(v_old) + 0.5*g_old@self.model.M_coef@(v_new-v_old) \
                  + self.dt/4*g_old@self.model.M_coef@(v_old+v_new) + self.dt/4*temp1

        g_new = self.eva_grad_R(v_new)
        temp2 = self.prior.c_half@g_new
        temp2 = np.sum(temp2*(self.model.M_coef@temp2))
        rho_new_old = self.R(v_new) + 0.5*g_new@self.model.M_coef@(v_old-v_new) \
                  + self.dt/4*g_new@self.model.M_coef@(v_old+v_new) + self.dt/4*temp2

        return np.exp(min(0.0, rho_old_new - rho_new_old))

    def s_sample(self, v_old, K):
        for i in range(K):
            w = self.prior.generate_sample(flag='only_vec')
            a, c, b = np.sqrt(1-self.beta*self.beta), -2*self.dt/(2+self.dt), self.beta
            g_old = self.eva_grad_R(v_old)
            v_new = a*v_old + c*(self.prior.c@g_old) + b*w
            acc = self.accept_R(v_new, v_old, g_old)
            n = np.random.uniform(0, 1)
            if acc >= n:
                v_old = v_new.copy()
                # print(i)
                break
        return v_old

    def aa(self, u_new, u_old):
        if self.res_FMALA == True:
            g_old = self.model.eval_grad_residual(u_old)
            temp1 = (self.prior.c_half@g_old)
            temp1 = np.sum(temp1*(self.model.M_coef@temp1))
            gr_old = self.eva_grad_R(u_old)
            temp11 = self.prior.c_half@gr_old
            temp11 = np.sum(temp1*(self.model.M_coef@temp11))
            rho_old_new = self.phi(u_old) + 0.5*g_old@self.model.M_coef@(u_new-u_old) \
                      + self.dt/4*g_old@self.model.M_coef@(u_old+u_new) + self.dt/4*temp1 \
                      + self.dt/2*temp11

            g_new = self.model.eval_grad_residual(u_new)
            temp2 = self.prior.c_half@g_new
            temp2 = np.sum(temp2@(self.model.M_coef@temp2))
            gr_new = self.eva_grad_R(u_new)
            temp22 = self.prior.c_half@gr_new
            temp22 = np.sum(temp2*(self.model.M_coef@temp22))
            rho_new_old = self.phi(u_new) + 0.5*g_new@self.model.M_coef@(u_old-u_new) \
                      + self.dt/4*g_new@self.model.M_coef@(u_old+u_new) + self.dt/4*temp2 \
                      + self.dt/2*temp22
        elif self.res_FMALA == False:
            rho_old_new, rho_new_old = self.phi(u_old), self.phi(u_new)

        return np.exp(min(0.0, rho_old_new - rho_new_old))

    def generate_chain(self, length_total=1e5, K=20, callback=None):
        chain = []
        ac_rate = 0
        ac_num = 0
        _, u_old = self.prior.generate_sample()
        chain.append(u_old.copy())

        i = 1
        si, index = 0, 0
        while i <= length_total:
            u_new = self.s_sample(u_old, K=K)
            acc = self.aa(u_new, u_old)
            n = np.random.uniform(0, 1)
            if acc >= n:
                chain.append(u_new.copy())
                u_old = u_new.copy()
                ac_num += 1
            else:
                chain.append(u_old.copy())
            i += 1

            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_' + np.str(np.int(index)), chain)
                    del chain
                    chain = []
                    index += 1

            ac_rate = ac_num/i

            if callback is not None:
                callback([u_old, i, ac_rate])

        if self.path is None:
            return [chain, ac_rate]
        else:
            return [ac_rate, self.path, np.int(index)]

###################################################################
class HierarchicalMethod(object):
    '''
    Bayesian method for TV-Gaussian prior with hyper-parameters
    '''
    def __init__(self, prior_u, prior_lam, prior_tau, phi, R, trans_lam, \
                 trans_tau, data, beta=0.01, save_num=np.int(1e4), path=None):
        self.prior_u = prior_u
        self.prior_lam = prior_lam
        self.prior_tau = prior_tau
        self.R = R
        self.phi = phi
        self.trans_lam = trans_lam
        self.trans_tau = trans_tau
        self.beta = beta
        self.data = data
        self.save_num = save_num
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def generate_chain(self, length_total=1e5, callback=None, uk=None, lamI=None, tauI=None):
        chain_u, chain_lam, chain_tau = [], [], []
        if uk is None:
            _, uk = self.prior_u.generate_sample()
        else:
            uk = uk

        if lamI is None:
            lam = self.prior_lam.generate_sample()
        else:
            lam = lamI

        if tauI is None:
            tau = self.prior_tau.generate_sample()
        else:
            tau = tauI
        print('Initial tau = ', round(tau, 4))

        chain_u.append(uk.copy())
        if tau is not np.ndarray:
            chain_tau.append(tau)
            chain_lam.append(lam)
        else:
            chain_tau.append(tau.copy())
            chain_lam.append(lam.copy())

        ac_rate1, ac_rate2 = 0, 0
        ac_num1, ac_num2 = 0, 0

        p1 = self.prior_lam.eval_density
        p0 = self.prior_tau.eval_density

        def a1(u_new, u_old, lam_new, lam_old, tau):
            term1 = self.phi(u_old, tau, self.data) - self.phi(u_new, tau, self.data)
            #lam_old, lam_new = 500, 500
            term2 = lam_old*self.R(u_old) - lam_new*self.R(u_new)
            term3 = np.log(p1(lam_new)+1e-15) - np.log(p1(lam_old)+1e-15)
            #print(term1, term2, term3)
            return np.exp(min(0.0, term1 + term2 + term3))
            #return np.exp(min(0.0, term1 + term2))

        def a2(tau_new, tau_old, u):
            term1 = self.phi(u, tau_old, self.data) - self.phi(u, tau_new, self.data)

            if tau_old.ndim == 0:
                N_d = len(self.data)
                term2 = -0.5*N_d*(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term3 = (np.log(p0(tau_new)+1e-15) - np.log(p0(tau_old)+1e-15))
            else:
                term2 = -0.5*np.sum(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term3 = np.sum(np.log(p0(tau_new)+1e-15) - np.log(p0(tau_old)+1e-15))

            return np.exp(min(0.0, term1 + term2 + term3))
            #return np.exp(min(0.0, term2 + term3))

        i=1
        si, index = 0, 0
        while i <= length_total:
            a = np.sqrt(1-self.beta*self.beta)
            b = self.beta
            # generate u_new
            _, xik = self.prior_u.generate_sample()
            vk = a*uk + b*xik
            # generate lam_new
            lam_ = self.trans_lam.generate_sample(lam)

            t = a1(vk, uk, lam_, lam, tau)
            r = np.random.uniform(0, 1)
            if t >= r:
                chain_u.append(vk.copy())
                uk = vk.copy()
                chain_lam.append(lam_)
                lam = lam_
                ac_num1 += 1
            else:
                chain_u.append(uk.copy())
                chain_lam.append(lam)
            ac_rate1 = ac_num1/i

            # update the parameter tau
            tau_ = self.trans_tau.generate_sample(tau)

            t = a2(tau_, tau, uk)
            r = np.random.uniform(0, 1)
            if t >= r:
                if tau_.ndim == 0:
                    chain_tau.append(tau_)
                    tau = tau_
                else:
                    chain_tau.append(tau_.copy())
                    tau = tau_.copy()
                ac_num2 += 1
            else:
                if tau.ndim == 0:
                    chain_tau.append(tau)
                else:
                    chain_tau.append(tau.copy())
            ac_rate2 = ac_num2/i

            i += 1

            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_u_' + np.str(np.int(index)), chain_u)
                    np.save(self.path + 'sample_lam_' + np.str(np.int(index)), chain_lam)
                    np.save(self.path + 'sample_tau_' + np.str(np.int(index)), chain_tau)
                    del chain_u, chain_lam, chain_tau
                    chain_u, chain_lam, chain_tau = [], [], []
                    index += 1

            if callback is not None:
                callback([uk, i, ac_rate1, ac_rate2, np.mean(np.array(chain_tau)),\
                          np.mean(np.array(chain_lam))])

        if self.path is None:
            return [chain_u, chain_lam, chain_tau, ac_rate1, ac_rate2]
        else:
            return [ac_rate1, ac_rate2, self.path, np.int(index)]

###################################################################
class HierarchicalMethod2(object):
    '''
    Bayesian method for TV-Gaussian prior with hyper-parameters
    '''
    def __init__(self, prior_u, prior_lam, prior_tau, prior_u0, phi, R, trans_lam, \
                 trans_tau, data, beta=0.01, beta0=0.01, save_num=np.int(1e4), path=None):
        self.prior_u = prior_u
        self.prior_lam = prior_lam
        self.prior_tau = prior_tau
        self.prior_u0 = prior_u0
        self.R = R
        self.phi = phi
        self.trans_lam = trans_lam
        self.trans_tau = trans_tau
        self.beta = np.array(beta)
        self.beta0 = beta0
        self.data = np.array(data)
        self.CM_inner = self.prior_u0.evaluate_CM_inner
        self.save_num = save_num
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def generate_chain(self, length_total=1e5, callback=None, uk=None, u0k=None, \
                       lamI=None, tauI=None):
        chain_u, chain_lam, chain_tau, chain_tau_mean, chain_u0 = [], [], [], [], []
        if uk is None:
            uk = self.prior_u.generate_sample(flag='only_vec')
        else:
            uk = np.array(uk)

        if u0k is None:
            u0k = 0.0*self.prior_u0.generate_sample(flag='only_vec')
        else:
            u0k = np.array(u0k)

        if lamI is None:
            lam = self.prior_lam.generate_sample()
        else:
            lam = np.array(lamI)

        if tauI is None:
            tau = self.prior_tau.generate_sample()
        else:
            tau = np.array(tauI)

        #print('Initial tau = ', round(np.mean(tau), 4))

        chain_u.append(uk.copy())
        if tau is not np.ndarray:
            chain_tau.append(tau)
            chain_lam.append(lam)
        else:
            chain_tau.append(tau.copy())
            chain_lam.append(lam.copy())

        ac_rate1, ac_rate2, ac_rate3, ac_rate4 = 0, 0, 0, 0
        ac_num1, ac_num2, ac_num3, ac_num4 = 0, 0, 0, 0

        potential1 = self.prior_lam.eval_potential
        potential0 = self.prior_tau.eval_potential
        p0 = self.prior_tau.eval_density

        def a1(u_new, u_old, lam, tau, u0):
#            lam = lam**2
            term1 = self.phi(u_old, tau, self.data) - self.phi(u_new, tau, self.data)
            term2 = lam*self.R(u_old) - lam*self.R(u_new)
            term3 = self.CM_inner(u0, u_new) - self.CM_inner(u0, u_old)
            return np.exp(min(0.0, term1 + term2 + term3))

        def a2(lam_new, lam_old, u):
#            lam_old, lam_new = lam_old**2, lam_new**2
#            term1 = (lam_old**2 - lam_new**2)*self.R(u)
            term1 = (lam_old - lam_new)*self.R(u)
            term2 = potential1(lam_new) - potential1(lam_old)
            #print(lam_new, lam_old, '-----')
            #print(term1, term2)
            return np.exp(min(0.0, term1 + term2))

        def a3(tau_new, tau_old, u):
            tau_new, tau_old = np.array(tau_new), np.array(tau_old)
            term1 = self.phi(u, tau_old, self.data) - self.phi(u, tau_new, self.data)
            if tau_old.ndim == 0:
                N_d = len(self.data)
                term2 = -0.5*N_d*(np.log(tau_old) - np.log(tau_new))
                term3 = potential0(tau_new) - potential0(tau_old)
#                term3 = np.log(p0(tau_new)+1e-20) - np.log(p0(tau_old)+1e-20)
            else:
                term2 = -0.5*np.sum(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term3 = np.sum(potential0(tau_new) - potential0(tau_old))
#                term3 = np.sum(np.log(p0(tau_new)+1e-20) - np.log(p0(tau_old)+1e-20))

#            print(term1, term2, term3)
            return np.exp(min(0.0, term1 + term2 + term3))
#            return min(1, np.exp(term1)*(np.power(tau_new/tau_old, N_d/2))*p0(tau_new)/p0(tau_old))

        def a4(u0_new, u0_old, u):
            temp1 = 0.5*self.CM_inner(u0_new, u0_new) - self.CM_inner(u0_new, u)
            temp2 = 0.5*self.CM_inner(u0_old, u0_old) - self.CM_inner(u0_old, u)
            #print(temp1, temp2, np.exp(min(0.0, temp2 - temp1)))
            return np.exp(min(0.0, temp2 - temp1))

        i=1
        si, index = 0, 0
        #adjustL, N_adjust, adjust_flag = [0, 0, 0, 0], 2000, False
        #r_mean = []
        while i <= length_total:
            a = np.sqrt(1-self.beta*self.beta)
            b = self.beta
            # generate u_new
#            if i == 1:
#                for k in range(10000):
#                    _, xik = self.prior_u.generate_sample()
#                    vk = a*uk + b*xik
#
#                    t = a1(vk, uk, lam, tau, u0k)
#                    r = np.random.uniform(0, 1)
#                    if t >= r:
#                        uk = vk.copy()

            xik = self.prior_u.generate_sample(flag='only_vec')
            vk = a*uk + b*xik

            t = a1(vk, uk, lam, tau, u0k)
            r = np.random.uniform(0, 1)
            if t >= r:
                chain_u.append(vk.copy())
                uk = vk.copy()
                ac_num1 += 1
            else:
                chain_u.append(uk.copy())
            ac_rate1 = ac_num1/i

#            if i%2000 == 0:
#                self.prior_u.update_mean_fun(np.mean(chain_u))

            #generate u0
#            a = np.sqrt(1-self.beta0*self.beta0)
#            b = self.beta0
#            etak = self.prior_u0.generate_sample(flag='only_vec')
#            v0k = a*u0k + b*etak
#
#            t = a4(v0k, u0k, uk)
#            r = np.random.uniform(0, 1)
#            if t >= r:
#                chain_u0.append(v0k.copy())
#                u0k = v0k.copy()
#                ac_num4 += 1
#            else:
#                chain_u0.append(u0k.copy())
#            ac_rate4 = ac_num4/i

            #generate lam_new
#            lam, lam_ = np.array(200), np.array(200)
            lam_ = self.trans_lam.generate_sample(lam)
            t = a2(lam_, lam, uk)
            r = np.random.uniform(0, 1)
            if t >= r:
                chain_lam.append(lam_)
                lam = lam_
                ac_num2 += 1
            else:
                chain_lam.append(lam)
            ac_rate2 = ac_num2/i

#            tau_temp= tau
#            for k in range(100):
#                tau_ = self.trans_tau.generate_sample(tau)
#                t = a3(tau_, tau_temp, uk)
#                r = np.random.uniform(0, 1)
#                if t >= r:
#                    tau_temp = tau_
#
#            tau_ = self.trans_tau.generate_sample(tau_temp)
#            t = a3(tau_, tau_temp, uk)

#            tau = np.array(400)
            tau_ = self.trans_tau.generate_sample(tau)
            t = a3(tau_, tau, uk)
            r = np.random.uniform(0, 1)
            if t >= r:
                if tau_.ndim == 0:
                    chain_tau.append(tau_)
                    tau = tau_
                    chain_tau_mean.append(tau_)
                else:
                    chain_tau.append(tau_.copy())
                    tau = tau_.copy()
                    chain_tau_mean.append(np.mean(tau_))
                ac_num3 += 1
            else:
                if tau.ndim == 0:
                    chain_tau.append(tau)
                    chain_tau_mean.append(tau)
                else:
                    chain_tau.append(tau.copy())
                    chain_tau_mean.append(np.mean(tau))
            ac_rate3 = ac_num3/i

            i += 1

            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_u_' + np.str(np.int(index)), chain_u)
                    np.save(self.path + 'sample_lam_' + np.str(np.int(index)), chain_lam)
                    np.save(self.path + 'sample_tau_' + np.str(np.int(index)), chain_tau)
                    np.save(self.path + 'sample_u0_' + np.str(np.int(index)), chain_u0)
                    del chain_u, chain_lam, chain_tau, chain_u0
                    chain_u, chain_lam, chain_tau, chain_u0 = [], [], [], []
                    index += 1

            if callback is not None:
                callback([uk, i, ac_rate1, ac_rate2, ac_rate3, ac_rate4, tau, lam])

        if self.path is None:
            return [chain_u, chain_lam, chain_tau, chain_u0, ac_rate1, ac_rate2, ac_rate3, ac_rate4]
        else:
            return [ac_rate1, ac_rate2, ac_rate3, ac_rate4, self.path, np.int(index)]

###################################################################
class HierarchicalMethodSplit2(object):
    '''
    Bayesian method for TV-Gaussian prior with hyper-parameters
    '''
    def __init__(self, prior_u, prior_lam, prior_tau, phi, R, trans_lam, \
                 trans_tau, data, beta=0.01, save_num=np.int(1e4), path=None):
        self.prior_u = prior_u
        self.u0 = self.prior_u.mean_fun.vector()[:].copy()
        self.prior_u.update_mean_fun(0.0)
        self.prior_lam = prior_lam
        self.prior_tau = prior_tau
        self.R = R
        self.phi = phi
        self.trans_lam = trans_lam
        self.trans_tau = trans_tau
        self.beta = np.array(beta)
        self.data = np.array(data)
        self.save_num = save_num
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def generate_chain(self, length_total=1e5, K=10, callback=None, uk=None, u0k=None, \
                       lamI=None, tauI=None):
        chain_u, chain_lam, chain_tau, chain_tau_mean = [], [], [], []
        if uk is None:
            uk = self.prior_u.generate_sample(flag='only_vec')
        else:
            uk = np.array(uk)

        if lamI is None:
            lam = self.prior_lam.generate_sample()
        else:
            lam = np.array(lamI)

        if tauI is None:
            tau = self.prior_tau.generate_sample()
        else:
            tau = np.array(tauI)

        chain_u.append(uk.copy())
        if tau is not np.ndarray:
            chain_tau.append(tau)
            chain_lam.append(lam)
        else:
            chain_tau.append(tau.copy())
            chain_lam.append(lam.copy())

        ac_rate1, ac_rate2, ac_rate3 = 0, 0, 0
        ac_num1, ac_num2, ac_num3 = 0, 0, 0

        potential1 = self.prior_lam.eval_potential
        potential0 = self.prior_tau.eval_potential
        len_data = len(self.data)

        def a11(u_new, u_old, tau):
            term1 = self.phi(u_old, tau, self.data) - self.phi(u_new, tau, self.data)
            return np.exp(min(0.0, term1))

        def a12(u_new, u_old, lam):
            term1 = self.R(u_old, lam) - self.R(u_new, lam)
            return np.exp(min(0.0, term1))

        def a2(lam_new, lam_old, u):
            term1 = self.R(u, lam_old) - self.R(u, lam_new)
            term2 = potential1(lam_new) - potential1(lam_old)
            # print(term1, term2)
            return np.exp(min(0.0, term1 + term2))

        def a3(tau_new, tau_old, u):
            tau_new, tau_old = np.array(tau_new), np.array(tau_old)
            term1 = self.phi(u, tau_old, self.data) - self.phi(u, tau_new, self.data)
            if tau_old.ndim == 0:
                N_d = len(self.data)
#                term2 = -0.5*N_d*(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term2 = -N_d*(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term3 = potential0(tau_new) - potential0(tau_old)
                term3 = len_data*term3
            else:
#                term2 = -0.5*np.sum(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term2 = -np.sum(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term3 = np.sum(potential0(tau_new) - potential0(tau_old))

            # print(term1, term2, term3)
            return np.exp(min(0.0, term1 + term2 + term3))

        i, iu = 1, 1
        si, index = 0, 0
        while i <= length_total:
#            if i%400 == 0:
#                print(tau, lam)
            if i == 1:
                iter_num_u = 1
                iter_num_tau = 1
                iter_num_lam = 1
            else:
                iter_num_u = 1
                iter_num_tau = 1
                iter_num_lam = 1

            iter_i = 1
            while iter_i <= iter_num_u:
                a = np.sqrt(1-self.beta*self.beta)
                b = self.beta
                vkk = uk.copy()
                for kk in range(K):
                    # generate u_new
                    xik = self.prior_u.generate_sample(flag='only_vec')
                    # sample with non-zero prior mean, a little bit different to the zero case
                    vk = self.u0 + a*(uk - self.u0) + b*xik
                    t = a12(vk, uk, lam)
                    r = np.random.uniform(0, 1)
                    if t >= r:
                        vkk = vk.copy()
                        break

                t = a11(vkk, uk, tau)
                r = np.random.uniform(0, 1)
                if t >= r:
                    chain_u.append(vkk.copy())
                    uk = vkk.copy()
                    ac_num1 += 1
                else:
                    chain_u.append(uk.copy())
                iu = iu + 1
                ac_rate1 = ac_num1/iu

                iter_i = iter_i + 1

            #generate lam_new
#            lam, lam_ = 30, 30
            if i%iter_num_lam == 0:
                lam_ = self.trans_lam.generate_sample(lam)
                t = a2(lam_, lam, uk)
                r = np.random.uniform(0, 1)
                if t >= r:
                    chain_lam.append(lam_)
                    lam = lam_
                    ac_num2 += 1
                else:
                    chain_lam.append(lam)
                ac_rate2 = ac_num2/i

            #generate tau_new
#            tau, tau_ = np.array(400), np.array(400)
            if i%iter_num_tau == 0:
                tau_ = self.trans_tau.generate_sample(tau)
                t = a3(tau_, tau, uk)
                r = np.random.uniform(0, 1)
                if t >= r:
                    if tau_.ndim == 0:
                        chain_tau.append(tau_)
                        tau = tau_
                        chain_tau_mean.append(tau_)
                    else:
                        chain_tau.append(tau_.copy())
                        tau = tau_.copy()
                        chain_tau_mean.append(np.mean(tau_))
                    ac_num3 += 1
                else:
                    if tau.ndim == 0:
                        chain_tau.append(tau)
                        chain_tau_mean.append(tau)
                    else:
                        chain_tau.append(tau.copy())
                        chain_tau_mean.append(np.mean(tau))
                ac_rate3 = ac_num3/i

            i += 1

            if callback is not None:
                callback([uk, i, ac_rate1, ac_rate2, ac_rate3, tau, lam, \
                          np.average(np.array(chain_tau))])

            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_u_' + np.str(np.int(index)), chain_u)
                    np.save(self.path + 'sample_lam_' + np.str(np.int(index)), chain_lam)
                    np.save(self.path + 'sample_tau_' + np.str(np.int(index)), chain_tau)
                    del chain_u, chain_lam, chain_tau
                    chain_u, chain_lam, chain_tau, chain_u0 = [], [], [], []
                    index += 1

        if self.path is None:
            return [chain_u, chain_lam, chain_tau, chain_u0, ac_rate1, ac_rate2, ac_rate3]
        else:
            return [ac_rate1, ac_rate2, ac_rate3, self.path, np.int(index)]
        
###################################################################
class HierarchicalMethodSplit(object):
    '''
    Bayesian method for TV-Gaussian prior with hyper-parameters
    '''
    def __init__(self, prior_u, prior_lam, prior_tau, phi, R, trans_lam, \
                 trans_tau, data, beta=0.01, save_num=np.int(1e4), path=None):
        self.prior_u = prior_u
        self.prior_lam = prior_lam
        self.prior_tau = prior_tau
        self.R = R
        self.phi = phi
        self.trans_lam = trans_lam
        self.trans_tau = trans_tau
        self.beta = beta
        self.data = data
        self.save_num = save_num
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        
    def generate_chain(self, length_total=1e5, callback=None, uk=None, lamI=None, tauI=None):
        chain_u, chain_lam, chain_tau = [], [], []
        if uk is None:
            _, uk = self.prior_u.generate_sample()
        else:
            uk = uk
            
        if lamI is None:
            lam = self.prior_lam.generate_sample()
        else:
            lam = lamI
        
        if tauI is None:
            tau = self.prior_tau.generate_sample()
        else:
            tau = tauI
        print('Initial tau = ', round(tau, 4))
        
        chain_u.append(uk.copy())
        if tau is not np.ndarray:
            chain_tau.append(tau)
            chain_lam.append(lam)
        else:
            chain_tau.append(tau.copy())
            chain_lam.append(lam.copy())
        
        ac_rate1, ac_rate2 = 0, 0
        ac_num1, ac_num2 = 0, 0
        
        p1 = self.prior_lam.eval_density
        p0 = self.prior_tau.eval_density
        
        def a11(u_new, u_old, lam_new, lam_old, tau):
            #lam_old, lam_new = 500, 500
            #term2 = lam_old*self.R(u_old) - lam_new*self.R(u_new)
            term2 = lam_old*self.R(u_old) - lam_new*self.R(u_old)
            term3 = np.log(p1(lam_new)+1e-15) - np.log(p1(lam_old)+1e-15)
            #print(term1, term2, term3)
            return np.exp(min(0.0, term2 + term3))
            #return np.exp(min(0.0, term1 + term2))
            
        def a12(u_new, u_old, tau, lam):
            term1 = self.phi(u_old, tau, self.data) - self.phi(u_new, tau, self.data)
            term2 = lam*self.R(u_old) - lam*self.R(u_new)
            return np.exp(min(0.0, term1 + term2))
        
        def a2(tau_new, tau_old, u):
            term1 = self.phi(u, tau_old, self.data) - self.phi(u, tau_new, self.data)
            
            if tau_old.ndim == 0:
                N_d = len(self.data)
                term2 = -0.5*N_d*(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term3 = (np.log(p0(tau_new)+1e-15) - np.log(p0(tau_old)+1e-15))
            else:
                term2 = -0.5*np.sum(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
                term3 = np.sum(np.log(p0(tau_new)+1e-15) - np.log(p0(tau_old)+1e-15))
                
            return np.exp(min(0.0, term1 + term2 + term3))
            #return np.exp(min(0.0, term2 + term3))
        
        i=1
        si, index = 0, 0
        while i <= length_total:
            a = np.sqrt(1-self.beta*self.beta)
            b = self.beta
            
            temp_u = uk.copy()
            for kk in range(10):
                # generate u_new 
                #_, xik = self.prior_u.generate_sample()
                #vk = a*uk + b*xik
                vk = uk.copy()
                # generate lam_new
                lam_ = self.trans_lam.generate_sample(lam)    
                t = a11(vk, temp_u, lam_, lam, tau)
                r = np.random.uniform(0, 1)
                if t >= r:
                    temp_lam = lam_
                    lam = lam_
                    temp_u = vk.copy()
                else:
                    temp_lam = lam
                    temp_u = uk.copy()
#            temp_lam = 100
#            _, xik = self.prior_u.generate_sample()
#            temp_u = a*uk + b*xik
            _, xik = self.prior_u.generate_sample()
            vk = a*uk + b*xik
            t = a12(vk, uk, tau, temp_lam)
            #t = a12(temp_u, uk, tau, temp_lam)
            r = np.random.uniform(0, 1)
            if t >= r:
                chain_u.append(temp_u.copy())
                uk = temp_u.copy()
                chain_lam.append(temp_lam)
                lam = temp_lam
                ac_num1 += 1
            else:
                chain_u.append(uk.copy())
                chain_lam.append(lam)
            ac_rate1 = ac_num1/i
            
            # update the parameter tau
            tau_ = self.trans_tau.generate_sample(tau)
            
            t = a2(tau_, tau, uk)
            r = np.random.uniform(0, 1)
            if t >= r:
                if tau_.ndim == 0:
                    chain_tau.append(tau_)
                    tau = tau_
                else:
                    chain_tau.append(tau_.copy())
                    tau = tau_.copy()
                ac_num2 += 1
            else:
                if tau.ndim == 0:
                    chain_tau.append(tau)
                else:
                    chain_tau.append(tau.copy())
            ac_rate2 = ac_num2/i
            
            i += 1
            
            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path + 'sample_u_' + np.str(np.int(index)), chain_u)
                    np.save(self.path + 'sample_lam_' + np.str(np.int(index)), chain_lam)
                    np.save(self.path + 'sample_tau_' + np.str(np.int(index)), chain_tau)
                    del chain_u, chain_lam, chain_tau
                    chain_u, chain_lam, chain_tau = [], [], []
                    index += 1
        
            if callback is not None:
                callback([uk, i, ac_rate1, ac_rate2, np.mean(np.array(chain_tau)),\
                          np.mean(np.array(chain_lam))])
        
        if self.path is None:
            return [chain_u, chain_lam, chain_tau, ac_rate1, ac_rate2]
        else:
            return [ac_rate1, ac_rate2, self.path, np.int(index)]






























