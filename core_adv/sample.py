#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:59:05 2019

@author: jjx323
"""
import numpy as np
from scipy.special import erfc
import fenics as fe
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.misc import printred
# from timeit import default_timer as timer

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
        
    def generate_chain(self, length_total=1e5, callback=None, uk=None):
        chain = []
        if uk is None:
            uk = self.prior.generate_sample(flag='only_vec')
        else:
            uk = uk
        chain.append(uk.copy())
        ac_rate = 0
        ac_num = 0
        
        def aa(u_new, u_old):
            #return min(1, np.exp(self.phi(u_old)-self.phi(u_new)))
#            print(self.phi(u_old)-self.phi(u_new))
            return np.exp(min(0.0, self.phi(u_old)-self.phi(u_new)))
        
        i=1
        si, index = 0, 0
        while i <= length_total:
            a = np.sqrt(1-self.beta*self.beta)
            b = self.beta
            xik = self.prior.generate_sample(flag='only_vec')
            vk = a*uk + b*xik
            t = aa(vk, uk)
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
    def __init__(self, prior, R, phi, beta=0.01, save_num=np.int(1e4), num_start=0.0, path=None):
        self.prior = prior
        self.R = R
        self.phi = phi
        self.beta = beta
        self.dt = (2*(2 - beta**2 - 2*np.sqrt(1 - beta**2)))/(beta**2)
        self.save_num = save_num
        self.num_start = num_start
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
        si, index = 0, np.int(self.num_start)
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
    def __init__(self, model, phi, beta=0.01, res_FMALA=True, save_num=np.int(1e4), num_start=0.0, path=None):
        fun_R = lambda x: model.lam*self.model.R.evaluate_R(x)
        super().__init__(model.ref_u, fun_R, phi, beta, save_num, num_start, path)
        self.eva_grad_R = lambda x: model.lam*model.R.evaluate_grad_R(x)
        self.model = model
        self.res_FMALA = res_FMALA

    def accept_R(self, v_new, v_old, g_old):
        temp1 = self.prior.c_half_times(g_old)
        temp1 = np.sum(temp1*(self.model.M_coef@temp1))
        rho_old_new = self.R(v_old) + 0.5*g_old@self.model.M_coef@(v_new-v_old) \
                  + self.dt/4*g_old@self.model.M_coef@(v_old+v_new) + self.dt/4*temp1

        g_new = self.eva_grad_R(v_new)
        temp2 = self.prior.c_half_times(g_new)
        temp2 = np.sum(temp2*(self.model.M_coef@temp2))
        rho_new_old = self.R(v_new) + 0.5*g_new@self.model.M_coef@(v_old-v_new) \
                  + self.dt/4*g_new@self.model.M_coef@(v_old+v_new) + self.dt/4*temp2

        return np.exp(min(0.0, rho_old_new - rho_new_old))

    def s_sample(self, v_old, K):
        for i in range(K):
            w = self.prior.generate_sample(flag='only_vec')
            a, c, b = np.sqrt(1-self.beta*self.beta), -2*self.dt/(2+self.dt), self.beta
            g_old = self.eva_grad_R(v_old)
            v_new = a*v_old + c*(self.prior.c_times(g_old)) + b*w
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
            temp1 = (self.prior.c_half_times(g_old))
            temp1 = np.sum(temp1*(self.model.M_coef@temp1))
            gr_old = self.eva_grad_R(u_old)
            temp11 = self.prior.c_half_times(gr_old)
            temp11 = np.sum(temp1*(self.model.M_coef@temp11))
            rho_old_new = self.phi(u_old) + 0.5*g_old@self.model.M_coef@(u_new-u_old) \
                      + self.dt/4*g_old@self.model.M_coef@(u_old+u_new) + self.dt/4*temp1 \
                      + self.dt/2*temp11

            g_new = self.model.eval_grad_residual(u_new)
            temp2 = self.prior.c_half_times(g_new)
            temp2 = np.sum(temp2@(self.model.M_coef@temp2))
            gr_new = self.eva_grad_R(u_new)
            temp22 = self.prior.c_half_times(gr_new)
            temp22 = np.sum(temp2*(self.model.M_coef@temp22))
            rho_new_old = self.phi(u_new) + 0.5*g_new@self.model.M_coef@(u_old-u_new) \
                      + self.dt/4*g_new@self.model.M_coef@(u_old+u_new) + self.dt/4*temp2 \
                      + self.dt/2*temp22
        elif self.res_FMALA == False:
            rho_old_new, rho_new_old = self.phi(u_old), self.phi(u_new)

        return np.exp(min(0.0, rho_old_new - rho_new_old))

    def generate_chain(self, length_total=1e5, K=20, callback=None, u=None):
        chain = []
        ac_rate = 0
        ac_num = 0
        if u is None:
            u_old = self.prior.generate_sample(flag='only_vec')
        else:
            u_old = u
        chain.append(u_old.copy())

        i = 1
        si, index = 0, np.int(self.num_start)
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
class ElasticNetHierarchical2:
    def __init__(self, model, residual, reg1, reg2, reg3, hyper1, hyper2, hyper3, hyper_noise, step, eig_beta, \
                 save_num=np.int(1e4), num_start=0, use_fmala=False, path=None):
        self.model = model
        self.geometric_dim = self.model.geometric_dim
        self.residual = residual
        self.reg1, self.reg2, self.reg3 = reg1, reg2, reg3  ## related to parameters lam and delta
        self.hyper1, self.hyper2, self.hyper3, self.hyper_noise = hyper1, hyper2, hyper3, hyper_noise
        self.step_u, self.step_reg1 = step['u'], step['lam']
        self.step_reg2, self.step_noise = step['delta'], step['tau']
        self.save_num, self.num_start = save_num, num_start
        self.use_fmala = use_fmala
        self.len_data = len(self.model.d)
        self.mean_prior_v = self.reg2.R.mean_fun.vector()[:].copy()
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        self.dt = (2*(2 - self.step_u**2 - 2*np.sqrt(1 - self.step_u**2)))/(self.step_u**2)
        self.a, self.b = np.sqrt(1-self.step_u*self.step_u), self.step_u
        self.c = -2*self.dt/(2+self.dt)
        self.path_u = self.path + 'u/'
        self.path_lam = self.path + 'lam/'
        self.path_delta = self.path + 'delta/'
        self.path_tau = self.path + 'tau/'
        self.path_v = self.path + 'v/'
        if os.path.exists(self.path_u) == False: os.mkdir(self.path_u)
        if os.path.exists(self.path_lam) == False: os.mkdir(self.path_lam)
        if os.path.exists(self.path_delta) == False: os.mkdir(self.path_delta)
        if os.path.exists(self.path_v) == False: os.mkdir(self.path_v)

        ## set values related to eigenvalues of Gaussian
        self.eig_Laplacian = np.power(self.reg2.R.eigLam.diagonal(), -2/self.reg2.R.s)  # the eigenvalues of the Laplacian
        self.eig_Laplacian.sort()
        if self.reg2.R.s - 0.5*self.geometric_dim - 1 < 0:
            print('\033[1;31m')
            print("The value s of Gaussian must be larger than 1+0.5*dimension of the space!")
            print('\033[0m')
        epsilon = (self.reg2.R.s - 0.5*self.geometric_dim - 1)/10000
        self.eig_sd2 = np.power(self.eig_Laplacian, self.reg2.R.s - 0.5*self.geometric_dim - epsilon)
        self.eig_sd2.sort()
#        self.eig_sd2 = np.power(self.eig_Laplacian, self.reg2.R.s)
        self.eig_alpha_s = np.power(self.eig_Laplacian, self.reg2.R.s)
        self.eig_alpha_s.sort()
        self.eig_beta = eig_beta.copy()
        self.eig_beta.sort()
        self.len_eig = min(len(self.eig_Laplacian), len(self.eig_beta))
        self.eig_judge = np.power(self.eig_beta[:self.len_eig], 0.5)/self.eig_sd2[:self.len_eig]
        self.Const = 1
        self.eig_judge = self.eig_judge
        self.sorted_eig_judge = self.eig_judge.copy()
        self.sorted_eig_judge.sort()
        self.M_max = np.sum(self.sorted_eig_judge > self.sorted_eig_judge[-1]/1e3)
        log_alpha_1toMmax = np.log(self.eig_Laplacian[:min(self.M_max, self.len_eig)])
        log_beta_1toMmax = np.log(self.eig_beta[:min(self.M_max, self.len_eig)])
        t_const = 0.5*self.reg2.R.s
        self.M_sum = (t_const*np.sum(log_alpha_1toMmax) - 0.5*np.sum(log_beta_1toMmax))
        
        self.eig_sd22 = np.power(self.eig_Laplacian, 0.5*self.reg2.R.s - 0.5*self.geometric_dim)
        self.eig_sd22.sort()
        self.eig_judge2 = np.power(self.eig_beta[:self.len_eig], 0.5)/self.eig_sd22[:self.len_eig]

        tt = self.reg2.R.s - self.geometric_dim/2 - epsilon
        self.exponent = 1.0/((tt-1)/self.geometric_dim + 0.5 + epsilon/2 + 1)
        self.con_M_ = -np.log(1-0.1)
        self.pow2 = np.power(2, (tt-1)/self.geometric_dim + 0.5 + epsilon/2)
        aa = np.power(np.arange(1, len(self.eig_Laplacian)+1), 2/self.geometric_dim)
        self.diff_c1 = max(self.eig_Laplacian/aa)
        self.diff_c2 = max(self.eig_beta/aa)
        self.diff_c3 = self.diff_c2/np.power(self.diff_c1, self.reg2.R.s)
        
        self.num_start_chain = 0
    
    def update_step_u(self, step):
        self.step_u = step
        self.dt = (2*(2 - self.step_u**2 - 2*np.sqrt(1 - self.step_u**2)))/(self.step_u**2)
        self.a, self.b = np.sqrt(1-self.step_u*self.step_u), self.step_u
        self.c = -2*self.dt/(2+self.dt)        

    def ac_u_res(self, u_new, u_old, tau, scale):
#        u_old, u_new = scale*v_old, scale*v_new
        tt1, Fu_old = self.residual.eva(u_old, tau)
        tt2, Fu_new = self.residual.eva(u_new, tau)
        term1 = tt1 - tt2
        # print(term1, self.residual.eval(u_old, tau), tau)
        return np.exp(min(0.0, term1)), Fu_old, Fu_new

    def ac_u_R(self, u_new, u_old, lam, lam2, scale, g_old, strategy):
        if self.use_fmala == False:
#            u_old, u_new = scale*v_old, scale*v_new
#            term1 = lam*(self.reg1.eval(u_old) - self.reg1.eval(u_new))
            term1 = lam*(self.reg1.eva(u_old) - self.reg1.eva(u_new))
            return np.exp(min(0.0, term1))
#            if strategy['lam2'] == True:
#                tt1 = 0.5*lam2*self.reg3.R.evaluate_CM_inner(u_old, u_old)
#                tt2 = 0.5*lam2*self.reg3.R.evaluate_CM_inner(u_new, u_new)
#                term2 = tt1 - tt2
#                return np.exp(min(0.0, term1 + term2))
#            else:
#                return np.exp(min(0.0, term1))
        else:
            pass
#            temp1 = self.reg2.R.c_half_times(g_old)
#            temp1 = np.sum(temp1*(self.model.M_coef@temp1))
#            rho_old_new = lam*scale*(self.reg1.eval(v_old) + 0.5*g_old@self.model.M_coef@(v_new-v_old) \
#                          + self.dt/4*g_old@self.model.M_coef@(v_old+v_new) + self.dt/4*temp1)
#
#            g_new = self.reg1.eval_grad(v_new)
#            temp2 = self.reg2.R.c_half_times(g_new)
#            temp2 = np.sum(temp2*(self.model.M_coef@temp2))
#            rho_new_old = lam*scale*(self.reg1.eval(v_new) + 0.5*g_new@self.model.M_coef@(v_old-v_new) \
#                      + self.dt/4*g_new@self.model.M_coef@(v_old+v_new) + self.dt/4*temp2)

#            return np.exp(min(0.0, rho_old_new - rho_new_old))

    def ac_lam_gamma(self, lam_new, lam_old, delta):
        old_val, new_val = 0, 0
        # self.eig_judge = self.eig_beta_half/self.eig_sd2[:self.len_eig]
        temp = (np.power(lam_old, 1)/delta)*self.eig_judge
        M = np.int(np.sum(temp >= 1))
        log_alpha_1toM = np.log(self.eig_Laplacian[:min(M, self.len_eig)])
        log_beta_1toM = np.log(self.eig_beta[:min(M, self.len_eig)])
        old_val += -M*np.log(lam_old)
        # t_const = 0.5*(self.reg2.R.s-self.geometric_dim/2-1e-10)
        t_const = 0.5*self.reg2.R.s
        old_val += (t_const*np.sum(log_alpha_1toM) - 0.5*np.sum(log_beta_1toM))
        # old_val += (self.reg2.R.s-1)/(self.geometric_dim)*np.sum(np.log(np.arange(M)+1))
        # cc=10
        # old_val += M*np.log(cc)
        M_old = M

        temp = (np.power(lam_new, 1)/delta)*self.eig_judge
        M = np.int(np.sum(temp >= 1))
        log_alpha_1toM = np.log(self.eig_Laplacian[:min(M, self.len_eig)])
        log_beta_1toM = np.log(self.eig_beta[:min(M, self.len_eig)])
        new_val += -M*np.log(lam_new)
        # t_const = 0.5*(self.reg2.R.s-self.geometric_dim/2-1e-10)
        t_const = 0.5*self.reg2.R.s
        new_val += (t_const*np.sum(log_alpha_1toM) - 0.5*np.sum(log_beta_1toM))
        # new_val += (self.reg2.R.s-1)/(self.geometric_dim)*np.sum(np.log(np.arange(M)+1))
        # new_val += M*np.log(cc)
        M_new = M

        term = old_val - new_val
        # print(old_val, new_val)
        return np.exp(min(0.0, term)), M_old, M_new
    
    def ac_lam(self, lam_new, lam_old, Ru, delta, i, kk):
        old_val, new_val = 0.0, 0.0
        potential1 = self.hyper1.eval_potential
        # self.eig_judge = self.eig_beta_half/self.eig_sd2[:self.len_eig]
        ## update: lam_old*R(u) - lam_new*R(u)
        tt1_term1 = lam_old*Ru
        tt2_term1 = lam_new*Ru
        term1 = tt1_term1 - tt2_term1
        ## update: log[p(lam_new)] - log[p(lam_old)]
        term2 = potential1(lam_new) - potential1(lam_old)
        ## update: Z(\lambda)
#        M_base = np.int(np.sum(1/delta*self.eig_judge >= 1))
#        temp = np.power(lam_old, 1)/np.sqrt(delta)/1.414*self.eig_judge2
#        M_old = min(np.int(np.sum(temp >= 1)), self.M_max)
#        M_old = max(np.int(np.sum(temp >= 1)), self.M_max)
        lam_old = max(lam_old, 0)
        M_old = np.int(np.power(lam_old/np.sqrt(delta)/self.con_M_/self.pow2, self.exponent))
        M_old = max(M_old, self.len_eig)
#        M_old = max(M_old, self.M_max)
#        temp = np.power(lam_new, 1)/np.sqrt(delta)/1.414*self.eig_judge2
#        M_new = min(np.int(np.sum(temp >= 1)), self.M_max) 
#        M_new = max(np.int(np.sum(temp >= 1)), self.M_max)
        lam_new = max(lam_new, 0)
        M_new = np.int(np.power(lam_new/np.sqrt(delta)/self.con_M_/self.pow2, self.exponent))
        M_new = max(M_new, self.len_eig)
#        M_new = max(M_new, self.M_max)
#        if kk == 199: print(M_old, M_new)
        if False:
            old_val = -0.5*(self.M_max+M_old)*np.log(lam_old) + 0.25*(self.M_max+M_old)
            new_val = -0.5*(self.M_max+M_new)*np.log(lam_new) + 0.25*(self.M_max+M_new) 
        else:
#            old_val = -self.M_max*np.log(lam_old) + (M_old/2)*(np.log(delta)-1.837)
#            new_val = -self.M_max*np.log(lam_new) + (M_old/2)*(np.log(delta)-1.837) 
#            log_alpha_1toM = np.log(self.eig_Laplacian[:min(M_old, self.len_eig)])
#            log_beta_1toM = np.log(self.eig_beta[:min(M_old, self.len_eig)])
#            old_val += (0.5*self.reg2.R.s*np.sum(log_alpha_1toM) - 0.5*np.sum(log_beta_1toM))
#            log_alpha_1toM = np.log(self.eig_Laplacian[:min(M_new, self.len_eig)])
#            log_beta_1toM = np.log(self.eig_beta[:min(M_new, self.len_eig)])
#            new_val += (0.5*self.reg2.R.s*np.sum(log_alpha_1toM) - 0.5*np.sum(log_beta_1toM))
#            self.eig_beta = np.power(np.arange(1, M_old), 2/self.geometric_dim)
#            temp1 = np.power(lam_old, 2)/delta*self.eig_beta
#            self.eig_alpha_s = np.power(np.arange(1, M_old), \
#                                        2*self.reg2.R.s/self.geometric_dim)
#            temp2 = 2*self.eig_alpha_s
            tt = np.power(np.arange(1, M_old), 2*(1-self.reg2.R.s)/self.geometric_dim)
            tt = tt*self.diff_c3
            tt = np.power(lam_old, 2)/delta/2*tt
#            temp1 = np.power(lam_old, 2)/delta*self.eig_beta[:min(M_old, self.len_eig)]
#            temp2 = 2*self.eig_alpha_s[:min(M_old, self.len_ett = np.power(lam_old, 2)/delta
#/2*np.power(np.arange(1, M_old), 2*(1-self.reg2.R.s)/self.geometric_dim)ig)]
#            tt = temp1/(temp2)
            tt2 = np.power(tt, 0.5)
            index1, index2 = tt2 <= 10,  tt2 > 10
            old_val += np.sum(tt[index1] + np.log(erfc(tt2[index1])))
#            print('1: ', old_val)
            sequ = 1 - 0.5/np.power(tt2[index2], 2) + 0.75/np.power(tt2[index2], 4)
            tt3 = np.log(sequ/tt2[index2]/1.77245)
            old_val += np.sum(tt3)
#            print((tt[index2]-tt2[index2])[0],  np.log(sequ/tt2[index2]/1.77245)[0])
            
#            self.eig_beta = np.power(np.arange(1, M_new), 2/self.geometric_dim)
#            temp1 = np.power(lam_old, 2)/delta*self.eig_beta
#            self.eig_alpha_s = np.power(np.arange(1, M_new), \
#                                        2*self.reg2.R.s/self.geometric_dim)
#            temp2 = 2*self.eig_alpha_s
#            temp1 = np.power(lam_new, 2)/delta*self.eig_beta[:min(M_new, self.len_eig)]
#            temp2 = 2*self.eig_alpha_s[:min(M_new, self.len_eig)]
#            tt = temp1/(temp2)
            tt = np.power(np.arange(1, M_new), 2*(1-self.reg2.R.s)/self.geometric_dim)
            tt = tt*self.diff_c3
            tt = np.power(lam_new, 2)/delta/2*tt
            tt2 = np.power(tt, 0.5)
            index1, index2 = tt2 <= 10, tt2 > 10
            new_val += np.sum(tt[index1] + np.log(erfc(tt2[index1])))
            sequ = 1 - 0.5/np.power(tt2[index2], 2) + 0.75/np.power(tt2[index2], 4)
            tt3 = np.log(sequ/tt2[index2]/1.77245)
            new_val += np.sum(tt3)
            
        term3 = (old_val - new_val)

#        if kk == 199: 
#            print("-"*50)
#            print("lam: ", lam_old, lam_new)
#            print("lamR: ", tt1_term1, tt2_term1, term1)
#            print("logZ: ", old_val, new_val, term3)
#            print("sum: ", tt1_term1+old_val, tt2_term1+new_val, term1+term3) 
#            print("-"*50)
        return np.exp(min(0.0, term1 + term2 + term3)), old_val, new_val

    def ac_lam2(self, lam2_new, lam2_old, vk, delta):
        tt1 = 0.5*lam2_old*self.reg3.R.evaluate_CM_inner(vk, vk)
        tt2 = 0.5*lam2_new*self.reg3.R.evaluate_CM_inner(vk, vk)
        term1 = tt1 - tt2
        
        potential1 = self.hyper3.eval_potential
        term2 = potential1(lam2_new) - potential1(lam2_old)
        
        M_old = np.sum(lam2_old*self.alpha2 > delta*self.eig_alpha_s)
        M_new = np.sum(lam2_new*self.alpha2 > delta*self.eig_alpha_s)
        tt1 = 0.5*(np.log(self.eig_alpha_s[:min(M_old, self.len_eig)]) + \
                   np.log(lam2_old*self.alpha2[:min(M_old, self.len_eig)] + \
                            self.eig_alpha_s[:min(M_old, self.len_eig)]))
        tt1 = np.sum(tt1)
        tt2 = 0.5*(np.log(self.eig_alpha_s[:min(M_new, self.len_eig)]) + \
                   np.log(lam2_new*self.alpha2[:min(M_new, self.len_eig)] + \
                            self.eig_alpha_s[:min(M_new, self.len_eig)]))
        tt2 = np.sum(tt2)
#        print(tt1, tt2)
        term3 = tt1 - tt2
        
        return np.exp(min(0.0, term1 + term2 + term3))

    def ac_delta(self, delta_new, delta_old, v, lam, tau, Rv, Fu, logZ_lam_old, i, kk, use_explicit):
        if use_explicit == False:
#            scale_old = 1/np.sqrt(delta_old)
#            tt11 = self.residual.eval_with_no_forward_solver(Fu, tau) 
#            tt12 = lam*scale_old*Rv + logZ_lam_old
#            tt1 = tt11 + tt12
#    #        tt1, _ = self.residual.eval(scale_old*v, tau)
#    #        tt1 = tt1 + lam*scale_old*Rv
#            scale_new = 1/np.sqrt(delta_new)
#            tt21, _ = self.residual.eva(scale_new*v, tau)
#            tt22 = lam*scale_new*Rv
#            
##            temp = np.power(lam, 1)/np.sqrt(delta_new)/1.414*self.eig_judge2
#    #        M_old = min(np.int(np.sum(temp >= 1)), self.M_max)
##            M_old = max(np.int(np.sum(temp >= 1)), self.M_max)
##            temp1 = np.power(lam, 2)*self.eig_beta[:min(M_old, self.len_eig)]
##            temp2 = 2*delta_new*self.eig_alpha_s[:min(M_old, self.len_eig)]
##            tt_ = temp1/(temp2)
#            aa = np.power(lam/np.sqrt(delta_new)/self.con_M_/self.pow2, self.exponent)
#            if np.isnan(aa): M_new = self.len_eig
#            else: M_new = max(np.int(aa), self.len_eig)
#            tt_ = np.power(np.arange(1, M_new), 2*(1-self.reg2.R.s)/self.geometric_dim)
#            tt_ = tt_*self.diff_c3
#            tt_ = np.power(lam, 2)/delta_new/2*tt_
#            tt2_ = np.power(tt_, 0.5)
#            index1, index2 = tt2_ <= 10,  tt2_ > 10
#            fir1 = np.sum(tt_[index1] + np.log(erfc(tt2_[index1])))
#    #            print('1: ', old_val)
#            sequ = 1 - 0.5/np.power(tt2_[index2], 2) + 0.75/np.power(tt2_[index2], 4)
#            tt3_ = np.log(sequ/tt2_[index2]/1.77245)
#            logZ_lam_new = fir1 + np.sum(tt3_)
#            tt22 += logZ_lam_new
#            
#            tt2 = tt21 + tt22
#            term1 = tt1 - tt2
#    #        term1 = tt11 - tt21
            
            scale_old = 1/np.sqrt(delta_old)
            tt11 = self.residual.eval_with_no_forward_solver(Fu, tau) 
            scale_new = 1/np.sqrt(delta_new)
            tt21, _ = self.residual.eva(scale_new*v, tau)
            term1 = tt11 - tt21

            potential1 = self.hyper2.eval_potential
            term2 = potential1(delta_new) - potential1(delta_old)
            
            final = term1 + term2
        else:
            scale_old = 1/np.sqrt(delta_old)
            scale_new = 1/np.sqrt(delta_new)
            temp = np.power(lam, 1)/np.sqrt(delta_new)/1.414*self.eig_judge2
    #        M_old = min(np.int(np.sum(temp >= 1)), self.M_max)
            M_old = max(np.int(np.sum(temp >= 1)), self.M_max)
            temp1 = np.power(lam, 2)*self.eig_beta[:min(M_old, self.len_eig)]
            temp2 = 2*delta_new*self.eig_alpha_s[:min(M_old, self.len_eig)]
            tt_ = temp1/(temp2)
            tt2_ = np.power(tt_, 0.5)
            index1, index2 = tt2_ <= 10,  tt2_ > 10
            fir1 = np.sum(tt_[index1] + np.log(erfc(tt2_[index1])))
    #            print('1: ', old_val)
            sequ = 1 - 0.5/np.power(tt2_[index2], 2) + 0.75/np.power(tt2_[index2], 4)
            tt3_ = np.log(sequ/tt2_[index2]/1.77245)
            logZ_lam_new = fir1 + np.sum(tt3_)
            term1 = lam*scale_old*Rv -lam*scale_new*Rv + logZ_lam_old - logZ_lam_new
            
            potential1 = self.hyper2.eval_potential
            term2 = potential1(delta_new) - potential1(delta_old)  
            
            final = term1 + term2

#        if i % 199 == 0: 
#            print("-"*50)
#            print('delta: ', delta_old, delta_new, delta_old-delta_new)
#            print('residual: ', tt11, tt21, tt11-tt21)
#            print('lamR: ', lam*scale_old*Rv, lam*scale_new*Rv, lam*scale_old*Rv-lam*scale_new*Rv)
#            print('logZ: ', logZ_lam_old, logZ_lam_new, logZ_lam_old-logZ_lam_new)
#            print('sum: ', tt11+lam*scale_old*Rv+logZ_lam_old, tt21+lam*scale_new*Rv+logZ_lam_new, final)
#            print("-"*50)
        return np.exp(min(0.0, final))

    def ac_tau(self, tau_new, tau_old, u):
        pass

    def generate_chain(self, length_total=1e5, K=10, callback=None, v_init=None, \
                       lam_init=None, lam2_init=None, delta_init=None, tau_init=None, strategy=None):
        if strategy is None:
            strategy = {'lam': False, 'lam2': False, 'delta': False, 'tau': False}

        chain_v, chain_lam, chain_lam2, chain_delta, chain_tau, chain_u = [], [], [], [], [], []
        if lam_init is None: lam = self.hyper1.generate_sample()
        else: lam = np.array(lam_init)
        if lam2_init is None: lam2 = self.hyper3.generate_sample()
        else: lam2 = np.array(lam2_init) 
        if delta_init is None: delta = self.hyper2.generate_sample()
        else: delta = np.array(delta_init)
        if tau_init is None: tau = self.hyper_noise.generate_sample()
        else: tau = np.array(tau_init)
        if v_init is None: uk = self.reg2.R.generate_sample(flag='only_vec')
        else: uk = np.array(v_init/np.sqrt(delta))

        chain_v.append(uk.copy()*np.sqrt(delta))
        ac_rate_v, ac_num_v = 0, 0
        if strategy['lam'] == True: chain_lam.append(lam)
        ac_rate_lam, ac_num_lam = 0, 0
        if strategy['lam2'] == True: chain_lam2.append(lam2)
        ac_rate_lam2, ac_num_lam2 = 0, 0
        if strategy['delta'] == True: chain_delta.append(delta)
        ac_rate_delta, ac_num_delta = 0, 0
        if strategy['tau'] == True: chain_tau.append(tau)
        ac_rate_tau, ac_num_tau = 0, 0

        mix_time_ = 50
        iter_num_v, iter_num_lam, iter_num_lam2, iter_num_delta, iter_num_tau \
                        = np.int(1), np.int(mix_time_), np.int(1), np.int(mix_time_), np.int(1)

        i, iv, ilam, ilam2, idelta, itau = 1, 1, 1, 1, 1, 1
        si, index = 0, np.int(self.num_start)
        c_matrix = self.reg2.R.c
        a, b, c = self.a, self.b, self.c
        Rv, Fu, M = None, None, 0
        M_new = np.sum(np.power(lam, 1)/np.sqrt(delta)*self.eig_judge >=1)
        M_old = M_new
        def adaptive_step_u(delta):
            if delta <= 0.01: self.step_u = 0.002
            elif delta <= 0.005: self.step_u = 0.001
            elif delta <= 0.001: self.step_u = 0.0004
            self.dt = (2*(2 - self.step_u**2 - 2*np.sqrt(1 - self.step_u**2)))/(self.step_u**2)
            self.a, self.b = np.sqrt(1-self.step_u*self.step_u), self.step_u
            self.c = -2*self.dt/(2+self.dt)
            return self.a, self.b, self.c
            
        eps_lam = 2e-5
        lam_t = lam
        beta_delta = self.hyper2.step_length
        beta1 = np.sqrt(1-beta_delta**2)
        ac_u, ivv = 0, 0
        ac_delta, idd = 0, 0
        ac_lam, ill = 0, 0 
        adaptive_num = 1e5
        while i <= length_total:
#            a, b, c = adaptive_step_u(delta)
            scale = 1/np.sqrt(delta)
#            self.reg2.R.update_alpha(delta)
            select = False
            if i%iter_num_v == 0:
                if lam >= eps_lam:
                    uk_ = uk.copy()
                    for kk in range(K):
                        xik = self.reg2.R.generate_sample(flag='only_vec')
                        if self.use_fmala == True:
                            g_old = lam*self.reg1.eval_grad(uk)
                            utk = scale*(self.mean_prior_v + a*(uk/scale - self.mean_prior_v) + c*(c_matrix@g_old) + b*xik)
                        else:
#                            utk = scale*(self.mean_prior_v + a*(uk/scale - self.mean_prior_v) + b*xik)
                            utk = scale*(a*uk/scale + b*xik)
                            g_old = 0.0
#                        if i%100 == 0: print(1/scale)
                        t = self.ac_u_R(utk, uk, lam, lam2, scale, g_old/lam, strategy)
                        r = np.random.uniform(0, 1)
                        if t >= r:
                            uk_ = utk.copy()
                            select = True
                            break
                ## for small lam, ignore the TV term
                else:
                    select = True
                    xik = self.reg2.R.generate_sample(flag='only_vec')
                    uk_ = scale*(self.mean_prior_v + a*(uk/scale - self.mean_prior_v) + b*xik)
                
                if select == True:
                    t, Fu_old, Fu_new = self.ac_u_res(uk_, uk, tau, scale)
                r = np.random.uniform(0, 1)
                if t >= r and select == True:
                    chain_u.append(uk_.copy())
                    uk = uk_.copy()
                    ac_num_v += 1
                    ac_u += 1
                    ## Fu needs the forward solver, record this value may avoid the calcuations in the sequal
                    Fu = Fu_new
                else:
                    chain_u.append(uk.copy())
                    ## Fu needs the forward solver, record this value may avoid the calcuations in the sequal
                    Fu = Fu_old
                Ru = self.reg1.eva(uk)   # calculate once here to avoid calculations in the sequal
                Rv = Ru*np.sqrt(delta)
                iv += 1
                ivv += 1
                ac_rate_v = ac_num_v/iv
                vk = uk.copy()/scale
                chain_v.append(vk)
                
            if i + self.num_start_chain <= adaptive_num:
                if i%1000 == 0:
                    ac_rate_u_l = ac_u/ivv
                    ac_u, ivv = 0, 0
                    if ac_rate_u_l < 0.2:
                        self.update_step_u(self.step_u*0.5)
                        printred("reset step of u (0.5), current acc_rate = ", ac_rate_u_l, end='; ')
                        printred("step_length = ", self.step_u)
                        a, b, c = self.a, self.b, self.c
                    elif ac_rate_u_l > 0.5:
                        self.update_step_u(self.step_u*2.0)
                        printred("reset step of u (2.0), current acc_rate = ", ac_rate_u_l, end='; ')
                        printred("step_length = ", self.step_u)
                        a, b, c = self.a, self.b, self.c

            if strategy['lam'] == True:
                if i%iter_num_lam == 0:
#                    self.hyper1.beta = beta0*np.sqrt(delta)
                    for kk in range(mix_time_):
#                        if self.hyper1.dis == None:#'Gamma':
#                            lam_ = np.random.gamma(shape=(self.hyper1.alpha), \
#                                                   scale=1/(self.hyper1.beta+scale*Rv))
#                            lam_ += self.hyper1.step_length*lam_
#                            t, M_old, M_new = self.ac_lam_gamma(lam_, lam, delta)
#                        else:
                        propose = self.hyper1.generate_next_sample(np.array(0.0))
#                        lam_t_ = lam_t + self.hyper1.step_length*propose
                        lam_ = lam + self.hyper1.step_length*propose
#                        print(self.hyper1.step_length*propose)
#                        if kk == 199: 
#                            print("-"*50)
#                            print('lam (new; old): ', lam_t_, lam_t, propose)
#                            print('alpha, beta: ', self.hyper1.alpha, self.hyper1.beta)
#                            print("-"*50)
                        if lam_ <= 0:
                            chain_lam.append(lam)
                            logZ_lam = 1e-10
                        else:
                            t, logZ_old, logZ_new = self.ac_lam(lam_, lam, Ru, delta, i, kk)
#                            t, logZ_old, logZ_new = self.ac_lam(lam_t_, lam_t, Ru, 1, i)
                            r = np.random.uniform(0, 1)
                            if t >= r:
#                                lam_t = lam_t_
                                lam = lam_
                                ac_num_lam += 1
                                ac_lam += 1
#                                lam = lam_t*np.sqrt(delta)
                                chain_lam.append(lam)
                                logZ_lam = logZ_new
    #                            M = M_new
                            else:
                                chain_lam.append(lam)
                                logZ_lam = logZ_old
    #                            M = M_old
                        ilam += 1
                        ill += 1
                        ac_rate_lam = ac_num_lam/ilam
            else:
                logZ_lam = 1e-5
            
            if i + self.num_start_chain <= adaptive_num:
                if i%1000 == 0:
                    ac_rate_lam_l = ac_lam/ill
                    ac_lam, ill = 0, 0
                    if ac_rate_lam_l < 0.2:
                        self.hyper1.step_length = 0.5*self.hyper1.step_length
                        self.hyper1.cov_proposal = 0.5*self.hyper1.cov_proposal
                        printred("reset step of lam (0.5), current acc_rate = ", ac_rate_lam_l, end='; ')
                        printred("step_length = ", self.hyper1.step_length, end='; ')
                        printred("cov_prop = ", self.hyper1.cov_proposal)
                    elif ac_rate_lam_l > 0.5:
                        self.hyper1.step_length = 2*self.hyper1.step_length
                        self.hyper1.cov_proposal = 2*self.hyper1.cov_proposal
                        printred("reset step of lam (2.0), current acc_rate = ", ac_rate_lam_l, end='; ')
                        printred("step_length = ", self.hyper1.step_length, end='; ')
                        printred("cov_prop = ", self.hyper1.cov_proposal)
                    
            if strategy['lam2'] == True:
                if i%iter_num_lam2 == 0:
                    lam2_ = self.hyper3.generate_next_sample(lam2)
                    if lam2_ <= 0:
                        chain_lam2.append(lam2)
                    else:
                        t = self.ac_lam2(lam2_, lam2, vk, delta)
                        r = np.random.uniform(0, 1)
                        if t >= r:
                            chain_lam2.append(lam2_)
                            lam2 = lam2_
                            ac_num_lam2 += 1
                        else:
                            chain_lam2.append(lam2)
                    ilam2 += 1
                    ac_rate_lam2 = ac_num_lam2/ilam2

            if strategy['delta'] == True:
                if i%iter_num_delta == 0:
                    if i<0: use_explicit, numkk =True, 10
                    else: use_explicit, numkk = False, mix_time_                          
                    for kk in range(numkk):
                        if use_explicit:
                            cov_pro_delta = 1/(tau*delta*np.sum(Fu*Fu)+1e-15)
#                            print(cov_pro_delta)
                            term1 = tau*np.sqrt(delta)*np.sum(Fu*self.model.d)
#                            if strategy['lam'] == True: term2 = lam*Rv
#                            else: term2 = 0.0 
#                            mean_pro_delta = cov_pro_delta*(max(term1 - term2, 0.0))
                            mean_pro_delta = cov_pro_delta*(max(term1, 0.0))
                            prop = np.power(mean_pro_delta + cov_pro_delta*np.random.normal(0, 1), -2)
                            delta_ = beta1*delta + beta_delta*prop 
    #                        if i%200 == 0: 
    #                            print(mean_pro_delta, np.power(mean_pro_delta,-2), delta_)
                        else:
                            delta_ = delta + self.hyper2.step_length*self.hyper2.generate_next_sample(np.array(0.0))
                        if delta_ <= 0:#1e-4:
                            chain_delta.append(delta)
                        else:
                            t = self.ac_delta(delta_, delta, vk, lam, tau, Rv, \
                                              Fu, logZ_lam, i, kk, use_explicit)
                            r = np.random.uniform(0, 1)
                            if t >= r:
                                chain_delta.append(delta_)
                                delta = delta_
                                ac_num_delta += 1
                                ac_delta += 1
                            else:
                                chain_delta.append(delta)
                        idelta += 1
                        idd += 1
                        ac_rate_delta = ac_num_delta/idelta
                        
            if i + self.num_start_chain <= adaptive_num:
                if i%1000 == 0:
                    ac_rate_delta_l = ac_delta/idd
                    ac_delta, idd = 0, 0
                    if ac_rate_delta_l < 0.2:
                        self.hyper2.step_length = 0.5*self.hyper2.step_length
                        self.hyper2.cov_proposal = 0.5*self.hyper2.cov_proposal
                        printred("reset step of delta (0.5), current acc_rate = ", ac_rate_delta_l, end='; ')
                        printred("step_length = ", self.hyper2.step_length, end='; ')
                        printred("cov_prop = ", self.hyper2.cov_proposal)
                    elif ac_rate_delta_l > 0.5:
                        self.hyper2.step_length = 2*self.hyper2.step_length
                        self.hyper2.cov_proposal = 2*self.hyper2.cov_proposal
                        printred("reset step of delta (2.0), current acc_rate = ", ac_rate_delta_l, end='; ')
                        printred("step_length = ", self.hyper2.step_length, end='; ')
                        printred("cov_prop = ", self.hyper2.cov_proposal)
#                    if i%200 == 0: print('delta (new; old): ', delta_, delta)

            if strategy['tau'] == True:
                if i%iter_num_tau == 0:
                    tau_ = self.hyper_noise.generate_sample(tau)
                    t = self.ac_tau(tau_, tau, chain_u[-1])
                    r = np.random.uniform(0, 1)
                    if t >= r:
                        if tau_.ndim == 0:
                            chain_tau.append(tau_)
                            tau = tau_
                        else:
                            chain_tau.append(tau.copy())
                            tau = tau_.copy()
                        ac_num_tau += 1
                    else:
                        if tau.ndim == 0: chain_tau.append(tau)
                        else: chain_tau.append(tau.copy())
                    itau += 1
                    ac_rate_tau = ac_num_tau/itau

            if callback is not None:
                callback((1/np.sqrt(delta)*vk, lam, lam2, delta, tau), \
                         (ac_rate_v, ac_rate_lam, ac_rate_lam2, ac_rate_delta, ac_rate_tau), \
                         (i, iv, ilam, ilam2, idelta, itau))

            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path_u + 'sample_u_' + np.str(np.int(index)), chain_u)
                    if strategy['lam'] == True:
                        np.save(self.path_lam + 'sample_lam_' + np.str(np.int(index)), chain_lam)
                    if strategy['lam2'] == True:
                        np.save(self.path_lam + 'sample_lam2_' + np.str(np.int(index)), chain_lam)
                    if strategy['delta'] == True:
                        np.save(self.path_v + 'sample_v_' + np.str(np.int(index)), chain_v)
                        np.save(self.path_delta + 'sample_delta_' + np.str(np.int(index)), chain_delta)
                    if strategy['tau'] == True:
                        np.save(self.path_tau + 'sample_tau_' + np.str(np.int(index)), chain_tau)
                    chain_v, chain_u, chain_lam, chain_lam2, chain_delta, chain_tau = [], [], [], [], [], []
                    index += 1

            i += 1

        return self.path, (ac_rate_v, ac_rate_lam, ac_rate_lam2, ac_rate_delta, ac_rate_tau)

###################################################################
class ElasticNetHierarchical:
    def __init__(self, model, residual, reg1, reg2, reg3, hyper1, hyper2, hyper3, hyper_noise, step, eig_beta, \
                 save_num=np.int(1e4), num_start=0, use_fmala=False, path=None):
        self.model = model
        self.geometric_dim = self.model.geometric_dim
        self.residual = residual
        self.reg1, self.reg2, self.reg3 = reg1, reg2, reg3  ## related to parameters lam and delta
        self.hyper1, self.hyper2, self.hyper3, self.hyper_noise = hyper1, hyper2, hyper3, hyper_noise
        self.step_u, self.step_reg1 = step['u'], step['lam']
        self.step_reg2, self.step_noise = step['delta'], step['tau']
        self.save_num, self.num_start = save_num, num_start
        self.use_fmala = use_fmala
        self.len_data = len(self.model.d)
        self.mean_prior_v = self.reg2.R.mean_fun.vector()[:].copy()
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        self.dt = (2*(2 - self.step_u**2 - 2*np.sqrt(1 - self.step_u**2)))/(self.step_u**2)
        self.a, self.b = np.sqrt(1-self.step_u*self.step_u), self.step_u
        self.c = -2*self.dt/(2+self.dt)
        self.path_u = self.path + 'u/'
        self.path_lam = self.path + 'lam/'
        self.path_delta = self.path + 'delta/'
        self.path_tau = self.path + 'tau/'
        self.path_v = self.path + 'v/'
        if os.path.exists(self.path_u) == False: os.mkdir(self.path_u)
        if os.path.exists(self.path_lam) == False: os.mkdir(self.path_lam)
        if os.path.exists(self.path_delta) == False: os.mkdir(self.path_delta)
        if os.path.exists(self.path_v) == False: os.mkdir(self.path_v)

        ## set values related to eigenvalues of Gaussian
        self.eig_Laplacian = np.power(self.reg2.R.eigLam.diagonal(), -2/self.reg2.R.s)  # the eigenvalues of the Laplacian
        self.eig_Laplacian.sort()
        if self.reg2.R.s - 0.5*self.geometric_dim - 1 < 0:
            print('\033[1;31m')
            print("The value s of Gaussian must be larger than 1+0.5*dimension of the space!")
            print('\033[0m')
        epsilon = (self.reg2.R.s - 0.5*self.geometric_dim - 1)/10000
        self.eig_sd2 = np.power(self.eig_Laplacian, self.reg2.R.s - 0.5*self.geometric_dim - epsilon)
        self.eig_sd2.sort()
#        self.eig_sd2 = np.power(self.eig_Laplacian, self.reg2.R.s)
        self.eig_alpha_s = np.power(self.eig_Laplacian, self.reg2.R.s)
        self.eig_alpha_s.sort()
        self.eig_beta = eig_beta.copy()
        self.eig_beta.sort()
        self.len_eig = min(len(self.eig_Laplacian), len(self.eig_beta))
        self.eig_judge = np.power(self.eig_beta[:self.len_eig], 0.5)/self.eig_sd2[:self.len_eig]
        self.Const = 1
        self.eig_judge = self.eig_judge
        self.sorted_eig_judge = self.eig_judge.copy()
        self.sorted_eig_judge.sort()
        self.M_max = np.sum(self.sorted_eig_judge > self.sorted_eig_judge[-1]/1e3)
        log_alpha_1toMmax = np.log(self.eig_Laplacian[:min(self.M_max, self.len_eig)])
        log_beta_1toMmax = np.log(self.eig_beta[:min(self.M_max, self.len_eig)])
        t_const = 0.5*self.reg2.R.s
        self.M_sum = (t_const*np.sum(log_alpha_1toMmax) - 0.5*np.sum(log_beta_1toMmax))
        
        self.eig_sd22 = np.power(self.eig_Laplacian, 0.5*self.reg2.R.s - 0.5*self.geometric_dim)
        self.eig_sd22.sort()
        self.eig_judge2 = np.power(self.eig_beta[:self.len_eig], 0.5)/self.eig_sd22[:self.len_eig]

    def ac_u_res(self, v_new, v_old, tau, scale):
        u_old, u_new = scale*v_old, scale*v_new
        tt1, Fu_old = self.residual.eva(u_old, tau)
        tt2, Fu_new = self.residual.eva(u_new, tau)
        term1 = tt1 - tt2
        # print(term1, self.residual.eval(u_old, tau), tau)
        return np.exp(min(0.0, term1)), Fu_old, Fu_new

    def ac_u_R(self, v_new, v_old, lam, lam2, scale, g_old, strategy):
        if self.use_fmala == False:
            u_old, u_new = scale*v_old, scale*v_new
            term1 = lam*(self.reg1.eva(u_old) - self.reg1.eva(u_new))
            if strategy['lam2'] == True:
                tt1 = 0.5*lam2*self.reg3.R.evaluate_CM_inner(u_old, u_old)
                tt2 = 0.5*lam2*self.reg3.R.evaluate_CM_inner(u_new, u_new)
                term2 = tt1 - tt2
                return np.exp(min(0.0, term1 + term2))
            else:
                return np.exp(min(0.0, term1))
        else:
            temp1 = self.reg2.R.c_half_times(g_old)
            temp1 = np.sum(temp1*(self.model.M_coef@temp1))
            rho_old_new = lam*scale*(self.reg1.eva(v_old) + 0.5*g_old@self.model.M_coef@(v_new-v_old) \
                          + self.dt/4*g_old@self.model.M_coef@(v_old+v_new) + self.dt/4*temp1)

            g_new = self.reg1.eval_grad(v_new)
            temp2 = self.reg2.R.c_half_times(g_new)
            temp2 = np.sum(temp2*(self.model.M_coef@temp2))
            rho_new_old = lam*scale*(self.reg1.eva(v_new) + 0.5*g_new@self.model.M_coef@(v_old-v_new) \
                      + self.dt/4*g_new@self.model.M_coef@(v_old+v_new) + self.dt/4*temp2)

            return np.exp(min(0.0, rho_old_new - rho_new_old))

    def ac_lam_gamma(self, lam_new, lam_old, delta):
        old_val, new_val = 0, 0
        # self.eig_judge = self.eig_beta_half/self.eig_sd2[:self.len_eig]
        temp = (np.power(lam_old, 1)/delta)*self.eig_judge
        M = np.int(np.sum(temp >= 1))
        log_alpha_1toM = np.log(self.eig_Laplacian[:min(M, self.len_eig)])
        log_beta_1toM = np.log(self.eig_beta[:min(M, self.len_eig)])
        old_val += -M*np.log(lam_old)
        # t_const = 0.5*(self.reg2.R.s-self.geometric_dim/2-1e-10)
        t_const = 0.5*self.reg2.R.s
        old_val += (t_const*np.sum(log_alpha_1toM) - 0.5*np.sum(log_beta_1toM))
        # old_val += (self.reg2.R.s-1)/(self.geometric_dim)*np.sum(np.log(np.arange(M)+1))
        # cc=10
        # old_val += M*np.log(cc)
        M_old = M

        temp = (np.power(lam_new, 1)/delta)*self.eig_judge
        M = np.int(np.sum(temp >= 1))
        log_alpha_1toM = np.log(self.eig_Laplacian[:min(M, self.len_eig)])
        log_beta_1toM = np.log(self.eig_beta[:min(M, self.len_eig)])
        new_val += -M*np.log(lam_new)
        # t_const = 0.5*(self.reg2.R.s-self.geometric_dim/2-1e-10)
        t_const = 0.5*self.reg2.R.s
        new_val += (t_const*np.sum(log_alpha_1toM) - 0.5*np.sum(log_beta_1toM))
        # new_val += (self.reg2.R.s-1)/(self.geometric_dim)*np.sum(np.log(np.arange(M)+1))
        # new_val += M*np.log(cc)
        M_new = M

        term = old_val - new_val
        # print(old_val, new_val)
        return np.exp(min(0.0, term)), M_old, M_new

    def ac_lam(self, lam_new, lam_old, Ru, delta, i):
        old_val, new_val = 0.0, 0.0
        potential1 = self.hyper1.eval_potential
        # self.eig_judge = self.eig_beta_half/self.eig_sd2[:self.len_eig]
        ## update: lam_old*R(u) - lam_new*R(u)
        tt1_term1 = lam_old*Ru
        tt2_term1 = lam_new*Ru
        term1 = tt1_term1 - tt2_term1
        ## update: log[p(lam_new)] - log[p(lam_old)]
        term2 = potential1(lam_new) - potential1(lam_old)
        ## update: Z(\lambda)
#        M_base = np.int(np.sum(1/delta*self.eig_judge >= 1))
        temp = np.power(lam_old, 1)/np.sqrt(delta)/1.414*self.eig_judge2
#        M_old = min(np.int(np.sum(temp >= 1)), self.M_max)
        M_old = max(np.int(np.sum(temp >= 1)), self.M_max)
#        M_old = max(M_old, self.M_max)
        temp = np.power(lam_new, 1)/np.sqrt(delta)/1.414*self.eig_judge2
#        M_new = min(np.int(np.sum(temp >= 1)), self.M_max) 
        M_new = max(np.int(np.sum(temp >= 1)), self.M_max)
#        M_new = max(M_new, self.M_max)
        if False:
            old_val = -0.5*(self.M_max+M_old)*np.log(lam_old) + 0.25*(self.M_max+M_old)*(np.log(delta))
            new_val = -0.5*(self.M_max+M_new)*np.log(lam_new) + 0.25*(self.M_max+M_new)*(np.log(delta)) 
        else:
#            old_val = -self.M_max*np.log(lam_old) + (M_old/2)*(np.log(delta)-1.837)
#            new_val = -self.M_max*np.log(lam_new) + (M_old/2)*(np.log(delta)-1.837) 
#            log_alpha_1toM = np.log(self.eig_Laplacian[:min(M_old, self.len_eig)])
#            log_beta_1toM = np.log(self.eig_beta[:min(M_old, self.len_eig)])
#            old_val += (0.5*self.reg2.R.s*np.sum(log_alpha_1toM) - 0.5*np.sum(log_beta_1toM))
#            log_alpha_1toM = np.log(self.eig_Laplacian[:min(M_new, self.len_eig)])
#            log_beta_1toM = np.log(self.eig_beta[:min(M_new, self.len_eig)])
#            new_val += (0.5*self.reg2.R.s*np.sum(log_alpha_1toM) - 0.5*np.sum(log_beta_1toM))
            temp1 = np.power(lam_old, 2)*self.eig_beta[:min(M_old, self.len_eig)]
            temp2 = 2*delta*self.eig_alpha_s[:min(M_old, self.len_eig)]
            tt = temp1/(temp2)
            tt2 = np.power(tt, 0.5)
            index1, index2 = tt2 <= 10,  tt2 > 10
            old_val += np.sum(tt[index1] + np.log(erfc(tt2[index1])))
#            print('1: ', old_val)
            sequ = 1 - 0.5/np.power(tt2[index2], 2) + 0.75/np.power(tt2[index2], 4)
            tt3 = np.log(sequ/tt2[index2]/1.77245)
            old_val += np.sum(tt3)
#            print((tt[index2]-tt2[index2])[0],  np.log(sequ/tt2[index2]/1.77245)[0])
            
            temp1 = np.power(lam_new, 2)*self.eig_beta[:min(M_new, self.len_eig)]
            temp2 = 2*delta*self.eig_alpha_s[:min(M_new, self.len_eig)]
            tt = temp1/(temp2)
            tt2 = np.power(tt, 0.5)
            index1, index2 = tt2 <= 10, tt2 > 10
            new_val += np.sum(tt[index1] + np.log(erfc(tt2[index1])))
            sequ = 1 - 0.5/np.power(tt2[index2], 2) + 0.75/np.power(tt2[index2], 4)
            tt3 = np.log(sequ/tt2[index2]/1.77245)
            new_val += np.sum(tt3)
            
        term3 = (old_val - new_val)

#        if i%200 == 0: 
#            print("-"*50)
#            print("lam: ", lam_old, lam_new)
#            print("lamR: ", tt1_term1, tt2_term1, term1)
#            print("logZ: ", old_val, new_val, term3)
#            print("sum: ", tt1_term1+old_val, tt2_term1+new_val, term1+term3) 
#            print("-"*50)
        return np.exp(min(0.0, term1 + term2 + term3)), old_val, new_val    

    def ac_lam2(self, lam2_new, lam2_old, vk, delta):
        tt1 = 0.5*lam2_old*self.reg3.R.evaluate_CM_inner(vk, vk)
        tt2 = 0.5*lam2_new*self.reg3.R.evaluate_CM_inner(vk, vk)
        term1 = tt1 - tt2
        
        potential1 = self.hyper3.eval_potential
        term2 = potential1(lam2_new) - potential1(lam2_old)
        
        M_old = np.sum(lam2_old*self.alpha2 > delta*self.eig_alpha_s)
        M_new = np.sum(lam2_new*self.alpha2 > delta*self.eig_alpha_s)
        tt1 = 0.5*(np.log(self.eig_alpha_s[:min(M_old, self.len_eig)]) + \
                   np.log(lam2_old*self.alpha2[:min(M_old, self.len_eig)] + \
                            self.eig_alpha_s[:min(M_old, self.len_eig)]))
        tt1 = np.sum(tt1)
        tt2 = 0.5*(np.log(self.eig_alpha_s[:min(M_new, self.len_eig)]) + \
                   np.log(lam2_new*self.alpha2[:min(M_new, self.len_eig)] + \
                            self.eig_alpha_s[:min(M_new, self.len_eig)]))
        tt2 = np.sum(tt2)
#        print(tt1, tt2)
        term3 = tt1 - tt2
        
        return np.exp(min(0.0, term1 + term2 + term3))

    def ac_delta(self, delta_new, delta_old, v, lam, tau, Rv, Fu, logZ_lam_old, i):
        scale_old = 1/np.sqrt(delta_old)
        tt11 = self.residual.eval_with_no_forward_solver(Fu, tau) 
        tt12 = lam*scale_old*Rv + logZ_lam_old
        tt1 = tt11 + tt12
#        tt1, _ = self.residual.eval(scale_old*v, tau)
#        tt1 = tt1 + lam*scale_old*Rv
        scale_new = 1/np.sqrt(delta_new)
        tt21, _ = self.residual.eva(scale_new*v, tau)
        tt22 = lam*scale_new*Rv
        
        temp = np.power(lam, 1)/np.sqrt(delta_new)/1.414*self.eig_judge2
#        M_old = min(np.int(np.sum(temp >= 1)), self.M_max)
        M_old = max(np.int(np.sum(temp >= 1)), self.M_max)
        temp1 = np.power(lam, 2)*self.eig_beta[:min(M_old, self.len_eig)]
        temp2 = 2*delta_new*self.eig_alpha_s[:min(M_old, self.len_eig)]
        tt_ = temp1/(temp2)
        tt2_ = np.power(tt_, 0.5)
        index1, index2 = tt2_ <= 10,  tt2_ > 10
        fir1 = np.sum(tt_[index1] + np.log(erfc(tt2_[index1])))
#            print('1: ', old_val)
        sequ = 1 - 0.5/np.power(tt2_[index2], 2) + 0.75/np.power(tt2_[index2], 4)
        tt3_ = np.log(sequ/tt2_[index2]/1.77245)
        logZ_lam_new = fir1 + np.sum(tt3_)
        tt22 += logZ_lam_new
        
        tt2 = tt21 + tt22
        term1 = tt1 - tt2
#        term1 = tt12 - tt22

        potential1 = self.hyper2.eval_potential
        term2 = potential1(delta_new) - potential1(delta_old)

#        if i%200 == 0: 
#            print("-"*50)
#            print('delta: ', delta_old, delta_new)
#            print('residual: ', tt11, tt21)
#            print('lamR: ', lam*scale_old*Rv, lam*scale_new*Rv)
#            print('logZ: ', logZ_lam_old, logZ_lam_new)
#            print('sum: ', tt11+lam*scale_old*Rv+logZ_lam_old, tt21+lam*scale_new*Rv+logZ_lam_new)
#            print("-"*50)
        return np.exp(min(0.0, term1 + term2))
#        return np.exp(min(0.0, term2))

    def ac_tau(self, tau_new, tau_old, u):
        pass

    def generate_chain(self, length_total=1e5, K=10, callback=None, v_init=None, \
                       lam_init=None, lam2_init=None, delta_init=None, tau_init=None, strategy=None):
        if strategy is None:
            strategy = {'lam': False, 'lam2': False, 'delta': False, 'tau': False}

        chain_v, chain_lam, chain_lam2, chain_delta, chain_tau, chain_u = [], [], [], [], [], []
        if v_init is None: vk = self.reg2.R.generate_sample(flag='only_vec')
        else: vk = np.array(v_init)
        if lam_init is None: lam = self.hyper1.generate_sample()
        else: lam = np.array(lam_init)
        if lam2_init is None: lam2 = self.hyper3.generate_sample()
        else: lam2 = np.array(lam2_init) 
        if delta_init is None: delta = self.hyper2.generate_sample()
        else: delta = np.array(delta_init)
        if tau_init is None: tau = self.hyper_noise.generate_sample()
        else: tau = np.array(tau_init)

        chain_v.append(vk.copy())
        ac_rate_v, ac_num_v = 0, 0
        if strategy['lam'] == True: chain_lam.append(lam)
        ac_rate_lam, ac_num_lam = 0, 0
        if strategy['lam2'] == True: chain_lam2.append(lam2)
        ac_rate_lam2, ac_num_lam2 = 0, 0
        if strategy['delta'] == True: chain_delta.append(delta)
        ac_rate_delta, ac_num_delta = 0, 0
        if strategy['tau'] == True: chain_tau.append(tau)
        ac_rate_tau, ac_num_tau = 0, 0

        iter_num_v, iter_num_lam, iter_num_lam2, iter_num_delta, iter_num_tau \
                        = np.int(1), np.int(200), np.int(1), np.int(200), np.int(1)

        i, iv, ilam, ilam2, idelta, itau = 1, 1, 1, 1, 1, 1
        si, index = 0, np.int(self.num_start)
        c_matrix = self.reg2.R.c
        a, b, c = self.a, self.b, self.c
        Rv, Fu, M = None, None, 0
        M_new = np.sum(np.power(lam, 1)/delta*self.eig_judge >=1)
        M_old = M_new
        def adaptive_step_u(delta):
            if delta <= 0.01: self.step_u = 0.002
            elif delta <= 0.005: self.step_u = 0.001
            elif delta <= 0.001: self.step_u = 0.0004
            self.dt = (2*(2 - self.step_u**2 - 2*np.sqrt(1 - self.step_u**2)))/(self.step_u**2)
            self.a, self.b = np.sqrt(1-self.step_u*self.step_u), self.step_u
            self.c = -2*self.dt/(2+self.dt)
            return self.a, self.b, self.c
            
        eps_lam = 1e-5
        while i <= length_total:
#            a, b, c = adaptive_step_u(delta)
            scale = 1/np.sqrt(delta)
            if i%iter_num_v == 0:
                if lam >= eps_lam:
                    vk_ = vk.copy()
                    for kk in range(K):
                        xik = self.reg2.R.generate_sample(flag='only_vec')
                        if self.use_fmala == True:
                            g_old = lam*scale*self.reg1.eval_grad(vk)
                            vtk = self.mean_prior_v + a*(vk - self.mean_prior_v) + c*(c_matrix@g_old) + b*xik
                        else:
                            vtk = self.mean_prior_v + a*(vk - self.mean_prior_v) + b*xik
                            g_old = 0.0
                        t = self.ac_u_R(vtk, vk, lam, lam2, scale, g_old/lam/scale, strategy)
                        r = np.random.uniform(0, 1)
                        if t >= r:
                            vk_ = vtk.copy()
                            break
                ## for small lam, ignore the TV term
                else:
                    xik = self.reg2.R.generate_sample(flag='only_vec')
                    vk_ = self.mean_prior_v + a*(vk - self.mean_prior_v) + b*xik

                t, Fu_old, Fu_new = self.ac_u_res(vk_, vk, tau, scale)
                r = np.random.uniform(0, 1)
                if t >= r:
                    chain_v.append(vk_.copy())
                    vk = vk_.copy()
                    ac_num_v += 1
                    ## Fu needs the forward solver, record this value may avoid the calcuations in the sequal
                    Fu = Fu_new
                else:
                    chain_v.append(vk.copy())
                    ## Fu needs the forward solver, record this value may avoid the calcuations in the sequal
                    Fu = Fu_old
                Rv = self.reg1.eva(vk)   # calculate once here to avoid calculations in the sequal
                iv += 1
                ac_rate_v = ac_num_v/iv
                chain_u.append(scale*(vk.copy()))

            if strategy['lam'] == True:
                if i%iter_num_lam == 0:
                    for kk in range(200):
                        if self.hyper1.dis == None:#'Gamma':
                            lam_ = np.random.gamma(shape=(self.hyper1.alpha), \
                                                   scale=1/(self.hyper1.beta+scale*Rv))
                            lam_ += self.hyper1.step_length*lam_
                            t, M_old, M_new = self.ac_lam_gamma(lam_, lam, delta)
                        else:
                            lam_ = lam + self.hyper1.step_length*self.hyper1.generate_next_sample(np.array(0.0))
    #                    if i%200 == 0: print('lam (new; old): ', lam_, lam)
                        if lam_ <= 0:
                            chain_lam.append(lam)
                            logZ_lam = 0
                        else:
                            t, logZ_old, logZ_new = self.ac_lam(lam_, lam, scale*Rv, delta, i)
                            r = np.random.uniform(0, 1)
                            if t >= r:
                                chain_lam.append(lam_)
                                lam = lam_
                                ac_num_lam += 1
                                logZ_lam = logZ_new
    #                            M = M_new
                            else:
                                chain_lam.append(lam)
                                logZ_lam = logZ_old
    #                            M = M_old
                        ilam += 1
                        ac_rate_lam = ac_num_lam/ilam
                    
            if strategy['lam2'] == True:
                if i%iter_num_lam2 == 0:
                    lam2_ = self.hyper3.generate_next_sample(lam2)
                    if lam2_ <= 0:
                        chain_lam2.append(lam2)
                    else:
                        t = self.ac_lam2(lam2_, lam2, vk, delta)
                        r = np.random.uniform(0, 1)
                        if t >= r:
                            chain_lam2.append(lam2_)
                            lam2 = lam2_
                            ac_num_lam2 += 1
                        else:
                            chain_lam2.append(lam2)
                    ilam2 += 1
                    ac_rate_lam2 = ac_num_lam2/ilam2

            if strategy['delta'] == True:
                if i%iter_num_delta == 0:
                    for kk in range(200):
                        if False:
                            cov_pro_delta = 1/(tau*delta*np.sum(Fu*Fu)+1e-15)
                            term1 = tau*np.sqrt(delta)*np.sum(Fu*self.model.d)
                            if strategy['lam'] == True: term2 = lam*Rv
                            else: term2 = 0.0 
                            mean_pro_delta = cov_pro_delta*(max(term1 - term2, 0.0))
                            delta_ = np.power(mean_pro_delta + cov_pro_delta*np.random.normal(0, 1), -2)
    #                        if i%200 == 0: 
    #                            print(mean_pro_delta, np.power(mean_pro_delta,-2), delta_)
                        else:
                            delta_ = delta + self.hyper2.step_length*self.hyper2.generate_next_sample(np.array(0.0))
                        if delta_ <= 0.00:
                            chain_delta.append(delta)
                        else:
                            t = self.ac_delta(delta_, delta, vk, lam, tau, Rv, Fu, logZ_lam, i)
                            r = np.random.uniform(0, 1)
                            if t >= r:
                                chain_delta.append(delta_)
                                delta = delta_
                                ac_num_delta += 1
                            else:
                                chain_delta.append(delta)
                        idelta += 1
                        ac_rate_delta = ac_num_delta/idelta
#                    if i%200 == 0: print('delta (new; old): ', delta_, delta)

            if strategy['tau'] == True:
                if i%iter_num_tau == 0:
                    tau_ = self.hyper_noise.generate_sample(tau)
                    t = self.ac_tau(tau_, tau, chain_u[-1])
                    r = np.random.uniform(0, 1)
                    if t >= r:
                        if tau_.ndim == 0:
                            chain_tau.append(tau_)
                            tau = tau_
                        else:
                            chain_tau.append(tau.copy())
                            tau = tau_.copy()
                        ac_num_tau += 1
                    else:
                        if tau.ndim == 0: chain_tau.append(tau)
                        else: chain_tau.append(tau.copy())
                    itau += 1
                    ac_rate_tau = ac_num_tau/itau

            if callback is not None:
                callback((1/np.sqrt(delta)*vk, lam, lam2, delta, tau), \
                         (ac_rate_v, ac_rate_lam, ac_rate_lam2, ac_rate_delta, ac_rate_tau), \
                         (i, iv, ilam, ilam2, idelta, itau))

            if self.path is not None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path_u + 'sample_u_' + np.str(np.int(index)), chain_u)
                    if strategy['lam'] == True:
                        np.save(self.path_lam + 'sample_lam_' + np.str(np.int(index)), chain_lam)
                    if strategy['lam2'] == True:
                        np.save(self.path_lam + 'sample_lam2_' + np.str(np.int(index)), chain_lam)
                    if strategy['delta'] == True:
                        np.save(self.path_v + 'sample_v_' + np.str(np.int(index)), chain_v)
                        np.save(self.path_delta + 'sample_delta_' + np.str(np.int(index)), chain_delta)
                    if strategy['tau'] == True:
                        np.save(self.path_tau + 'sample_tau_' + np.str(np.int(index)), chain_tau)
                    chain_v, chain_u, chain_lam, chain_lam2, chain_delta, chain_tau = [], [], [], [], [], []
                    index += 1

            i += 1

        return self.path, (ac_rate_v, ac_rate_lam, ac_rate_lam2, ac_rate_delta, ac_rate_tau)

###################################################################
class HierarchicalMethodSplit(object):
    '''
    Bayesian method for TV-Gaussian prior with hyper-parameters
    '''
    def __init__(self, model, phi, eva_R, trans_lam, trans_tau, beta_u=0.01, beta_u0=0.01, \
                 save_num=np.int(1e4), num_start=0, use_fmala=False, path=None):
        self.model = model
        self.prior_u = self.model.prior_u

        self.eig_b = np.power(self.prior_u.eigLam.diagonal(), -2/self.prior_u.s)
        self.eig_b_u = np.power(self.eig_b, self.prior_u.s)
        self.eig_b_r = np.power(self.eig_b, 1.0)
        # self.hyper_lam = np.sqrt(self.eig_b_u/self.eig_b_r)
        self.hyper_lam = self.eig_b_u/self.eig_b_r

        self.mean_prior_u = self.prior_u.mean_fun.vector()[:].copy()
        # self.prior_u.update_mean_fun(0.0)
        self.prior_u_mean = self.model.prior_u_mean
        self.prior_lam = self.model.prior_lam
        self.prior_tau = self.model.prior_tau
        self.eva_R = eva_R
        self.eva_R.update(self.model.R)
        self.eva_grad_R = self.eva_R.grad_R
        self.R = self.eva_R.fun_R
        self.phi = phi
        self.trans_lam = trans_lam
        self.trans_tau = trans_tau
        self.beta = np.array(beta_u)
        self.beta_u0 = np.array(beta_u0)
        self.data = np.array(self.model.d)
        self.save_num = save_num
        self.num_start = num_start
        self.path = path
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

        self.dt = (2*(2 - self.beta**2 - 2*np.sqrt(1 - self.beta**2)))/(self.beta**2)
        self.a, self.b = np.sqrt(1-self.beta*self.beta), self.beta
        self.c = -2*self.dt/(2+self.dt)
        self.dt_u0 = (2*(2 - self.beta_u0**2 - 2*np.sqrt(1 - self.beta_u0**2)))/(self.beta_u0**2)
        self.a_u0, self.b_u0 = np.sqrt(1-self.beta_u0*self.beta_u0), self.beta_u0
        self.c_u0 = -2*self.dt_u0/(2+self.dt_u0)
        self.len_data = len(self.data)
        self.use_fmala = use_fmala
        self.path_u = self.path + 'u/'
        self.path_u0 = self.path + 'u0/'
        self.path_lam = self.path + 'lam/'
        self.path_tau =self.path + 'tau/'
        if os.path.exists(self.path_u) == False:
            os.mkdir(self.path_u)
        if os.path.exists(self.path_lam) == False:
            os.mkdir(self.path_lam)
        if os.path.exists(self.path_tau) == False:
            os.mkdir(self.path_tau)
        if os.path.exists(self.path_u0) == False:
            os.mkdir(self.path_u0)

    def ac_u_res(self, u_new, u_old, tau, u0):
        term1 = self.phi(u_old, tau, self.model) - self.phi(u_new, tau, self.model)
        # term2 = self.prior_u.evaluate_CM_inner(u_new, u0) - self.prior_u.evaluate_CM_inner(u_old, u0)
        # term2 = 0.5*self.prior_u.evaluate_CM_inner(u_new-u0, u_new-u0) \
        #         - 0.5*self.prior_u.evaluate_CM_inner(u_old-u0, u_old-u0)
        # return np.exp(min(0.0, term1 + term2))
        return np.exp(min(0.0, term1))

    def ac_u_R(self, u_new, u_old, lam, g_old, u0):
        if self.use_fmala == False:
            term1 = self.R(u_old, lam) - self.R(u_new, lam)
            # term1 = self.R(u_old - u0, lam) - self.R(u_new - u0, lam)
            return np.exp(min(0.0, term1))
        else:
            temp1 = self.prior_u.c_half_times(g_old)
            temp1 = np.sum(temp1*(self.model.M_coef@temp1))
            rho_old_new = self.R(u_old, lam) + 0.5*g_old@self.model.M_coef@(u_new-u_old) \
                          + self.dt/4*g_old@self.model.M_coef@(u_old+u_new) + self.dt/4*temp1
            # rho_old_new = self.R(u_old-u0, lam) + 0.5*g_old@self.model.M_coef@(u_new-u_old) \
            #               + self.dt/4*g_old@self.model.M_coef@(u_old+u_new-2*u0) + self.dt/4*temp1

            g_new = self.eva_grad_R(u_new, lam)
            # g_new = self.eva_grad_R(u_new-u0, lam)
            temp2 = self.prior_u.c_half_times(g_new)
            temp2 = np.sum(temp2*(self.model.M_coef@temp2))
            rho_new_old = self.R(u_new, lam) + 0.5*g_new@self.model.M_coef@(u_old-u_new) \
                      + self.dt/4*g_new@self.model.M_coef@(u_old+u_new) + self.dt/4*temp2
            # rho_new_old = self.R(u_new-u0, lam) + 0.5*g_new@self.model.M_coef@(u_old-u_new) \
            #           + self.dt/4*g_new@self.model.M_coef@(u_old+u_new-2*u0) + self.dt/4*temp2

            return np.exp(min(0.0, rho_old_new - rho_new_old))

    def ac_u0_R(self, u_new, u_old, lam, g_old, uu):
        if self.use_fmala == False:
            term1 = self.R(u_old, lam) - self.R(u_new, lam)
            # term1 = self.R(uu - u_old, lam) - self.R(uu - u_new, lam)
            return np.exp(min(0.0, term1))
        else:
            temp1 = self.prior_u_mean.c_half_times(g_old)
            temp1 = np.sum(temp1*(self.model.M_coef@temp1))
            rho_old_new = self.R(u_old, lam) + 0.5*g_old@self.model.M_coef@(u_new-u_old) \
                          + self.dt/4*g_old@self.model.M_coef@(u_old+u_new) + self.dt/4*temp1
            # rho_old_new = self.R(uu - u_old, lam) + 0.5*g_old@self.model.M_coef@(u_old-u_new) \
            #               + self.dt/4*g_old@self.model.M_coef@(2*uu-u_old-u_new) + self.dt/4*temp1

            g_new = self.eva_grad_R(u_new, lam)
            # g_new = self.eva_grad_R(uu-u_new, lam)
            temp2 = self.prior_u_mean.c_half_times(g_new)
            temp2 = np.sum(temp2*(self.model.M_coef@temp2))
            rho_new_old = self.R(u_new, lam) + 0.5*g_new@self.model.M_coef@(u_old-u_new) \
                      + self.dt/4*g_new@self.model.M_coef@(u_old+u_new) + self.dt/4*temp2
            rho_new_old = self.R(uu-u_new, lam) + 0.5*g_new@self.model.M_coef@(u_new-u_old) \
                      + self.dt/4*g_new@self.model.M_coef@(2*uu-u_old-u_new) + self.dt/4*temp2

            return np.exp(min(0.0, rho_old_new - rho_new_old))

    def ac_lam(self, lam_new, lam_old, u, u0):
        potential1 = self.prior_lam.eval_potential
        term1 = self.R(u, lam_old) - self.R(u, lam_new)
        # term1 = self.R(u-u0, lam_old) - self.R(u-u0, lam_new)
        term2 = potential1(lam_new) - potential1(lam_old)
        # term3 = self.R(u0, lam_new) - self.R(u0, lam_old)
        # print(term1, term2)
        # term3 = np.ceil(np.power(lam_new/3, 0.98))*np.log(lam_new) \
        #         - np.ceil(np.power(lam_old/3, 0.98))*np.log(lam_old)
        term3 = np.sum(self.hyper_lam <= lam_new/self.prior_u.alpha)*np.log(lam_new) \
                - np.sum(self.hyper_lam <= lam_old/self.prior_u.alpha)*np.log(lam_old)
        return np.exp(min(0.0, term1 + term2 + term3))

    def ac_tau(self, tau_new, tau_old, u):
        potential0 = self.prior_tau.eval_potential
        tau_new, tau_old = np.array(tau_new), np.array(tau_old)
        term1 = self.phi(u, tau_old, self.model) - self.phi(u, tau_new, self.model)
        if tau_old.ndim == 0:
            N_d = len(self.data)
            term2 = -0.5*N_d*(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
            # term2 = -N_d*(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
            term3 = potential0(tau_new) - potential0(tau_old)
            term3 = self.len_data*term3
        else:
            # term2 = -0.5*np.sum(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
            term2 = -np.sum(np.log(tau_old+1e-15) - np.log(tau_new+1e-15))
            term3 = np.sum(potential0(tau_new) - potential0(tau_old))

        # print(term1, term2, term3)
        return np.exp(min(0.0, term1 + term2 + term3))

    def ac_u0(self, u0_new, u0_old, u, lam):
        # term1 = 0.5*self.prior_u.evaluate_CM_inner(u0_new, u0_new) \
        #         - self.prior_u.evaluate_CM_inner(u0_new, u)
        # term2 = 0.5*self.prior_u.evaluate_CM_inner(u0_old, u0_old)  \
        #         - self.prior_u.evaluate_CM_inner(u0_old, u)
        # print(temp1, temp2, temp2-temp1, np.exp(min(0.0, temp2 - temp1)))
        term2 = 0.5*self.prior_u.evaluate_CM_inner(u0_old-u, u0_old-u)
        term1 = 0.5*self.prior_u.evaluate_CM_inner(u0_new-u, u0_new-u)
        # print(term2, term1, term2 - term1)
        # term3 = self.R(u0_new, lam) - self.R(u0_old, lam)
        # term3 = self.R(u - u0_old, lam) - self.R(u - u0_new, lam)
        # return np.exp(min(0.0, term2 - term1 + term3))
        return np.exp(min(0.0, term2 - term1))

    def generate_chain(self, length_total=1e5, K=10, callback=None, u_init=None, u0_init=None, \
                       lam_init=None, tau_init=None, strategy=None):
        if strategy is None:
            strategy = {'u0': True, 'lam': True, 'tau': True}

        chain_u, chain_u0, chain_lam, chain_tau, chain_tau_mean = [], [], [], [], []
        if u_init is None:
            uk = self.prior_u.generate_sample(flag='only_vec')
        else:
            uk = np.array(u_init)

        if u0_init is None:
            u0k = self.prior_u_mean.generate_sample(flag='only_vec')
        else:
            u0k = np.array(u0_init)

        if lam_init is None:
            lam = self.prior_lam.generate_sample()
        else:
            lam = np.array(lam_init)

        if tau_init is None:
            tau = self.prior_tau.generate_sample()
        else:
            tau = np.array(tau_init)

        chain_u.append(uk.copy())
        chain_u0.append(u0k.copy())
        if tau is not np.ndarray:
            chain_tau.append(tau)
            chain_lam.append(lam)
        else:
            chain_tau.append(tau.copy())
            chain_lam.append(lam.copy())

        ac_rate_u, ac_rate_u0, ac_rate_lam, ac_rate_tau = 0, 0, 0, 0
        ac_num_u, ac_num_u0, ac_num_lam, ac_num_tau = 0, 0, 0, 0

        i, iu = 1, 1
        si, index = 0, np.int(self.num_start)
        while i <= length_total:
#            if i%400 == 0:
#                print(tau, lam)
            if i == 1:
                iter_num_u = np.int(1)
                iter_num_tau = np.int(1)
                iter_num_lam = np.int(1)
            else:
                iter_num_u = np.int(1)
                iter_num_tau = np.int(1)
                iter_num_lam = np.int(1)

            iter_i = 1
            while iter_i <= iter_num_u:
                vkk = uk.copy()
                for kk in range(K):
                    ## generate u_new
                    xik = self.prior_u.generate_sample(flag='only_vec')
                    ## sample with non-zero prior mean, a little bit different to the zero case
                    if self.use_fmala == True:
                        g_old = self.eva_grad_R(uk, lam)
                        # g_old = self.eva_grad_R(uk - u0k, lam)
                        vk = self.mean_prior_u + self.a*(uk - self.mean_prior_u) + self.c*(self.prior_u.c@g_old) + self.b*xik
                        # vk = self.a*uk + self.c*(self.prior_u.c@g_old) + self.b*xik
                    else:
                        vk = self.mean_prior_u + self.a*(uk - self.mean_prior_u) + self.b*xik
                        # vk = self.a*uk + self.b*xik
                        g_old = None
                    t = self.ac_u_R(vk, uk, lam, g_old, u0k)
                    r = np.random.uniform(0, 1)
                    if t >= r:
                        vkk = vk.copy()
                        break

                t = self.ac_u_res(vkk, uk, tau, u0k)
                r = np.random.uniform(0, 1)
                if t >= r:
                    chain_u.append(vkk.copy())
                    uk = vkk.copy()
                    ac_num_u += 1
                else:
                    chain_u.append(uk.copy())
                iu = iu + 1
                ac_rate_u = ac_num_u/iu

                iter_i = iter_i + 1

            ## generate u0
            if strategy['u0'] == True:
                # if i == 1:
                #     ## using estimate from pre-sampling as start
                #     # u0k = uk.copy()
                #     u0k = self.prior_u_mean.mean_fun.vector()[:]
                # v0kk = u0k.copy()
                # for kk in range(K):
                #    etak = self.prior_u_mean.generate_sample(flag='only_vec')
                #    if self.use_fmala == True:
                #        g_old = self.eva_grad_R(uk-etak, lam)
                #        v0k = self.a_u0*u0k + self.c_u0*(self.prior_u_mean.c@g_old) + self.b_u0*etak
                #    else:
                #        v0k = self.a_u0*u0k + self.b_u0*etak
                #        g_old = None
                #    t = self.ac_u0_R(v0k, u0k, lam, g_old, uk)
                #    r = np.random.uniform(0, 1)
                #    if t >= r:
                #        v0kk = v0k.copy()
                #        break

                etak = self.prior_u_mean.generate_sample(flag='only_vec')
                v0kk = self.a_u0*u0k + self.b_u0*etak
                t = self.ac_u0(v0kk, u0k, uk, lam)
                r = np.random.uniform(0, 1)
                if t >= r:
                   chain_u0.append(v0kk.copy())
                   u0k = v0kk.copy()
                   ac_num_u0 += 1
                else:
                   chain_u0.append(u0k.copy())
                ac_rate_u0 = ac_num_u0/i

                self.mean_prior_u = u0k
                # self.prior_u.update_mean_fun(u0k.copy())

            ## generate lam_new
            if strategy['lam'] == True:
                if i%iter_num_lam == 0:
                    lam_ = self.trans_lam.generate_sample(lam)
                    t = self.ac_lam(lam_, lam, uk, u0k)
                    r = np.random.uniform(0, 1)
                    if t >= r:
                        chain_lam.append(lam_)
                        lam = lam_
                        ac_num_lam += 1
                    else:
                        chain_lam.append(lam)
                    ac_rate_lam = ac_num_lam/i

            ## generate tau_new
            if strategy['tau'] == True:
                if i%iter_num_tau == 0:
                    tau_ = self.trans_tau.generate_sample(tau)
                    t = self.ac_tau(tau_, tau, uk)
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
                        ac_num_tau += 1
                    else:
                        if tau.ndim == 0:
                            chain_tau.append(tau)
                            chain_tau_mean.append(tau)
                        else:
                            chain_tau.append(tau.copy())
                            chain_tau_mean.append(np.mean(tau))
                    ac_rate_tau = ac_num_tau/i

            if callback != None:
                callback((uk, u0k, lam, tau), (ac_rate_u, ac_rate_u0, ac_rate_lam, ac_rate_tau), i)

            if self.path != None:
                si += 1
                if np.int(si) == np.int(self.save_num):
                    si = 0
                    np.save(self.path_u + 'sample_u_' + np.str(np.int(index)), chain_u)
                    if strategy['lam'] == True:
                        np.save(self.path_lam + 'sample_lam_' + np.str(np.int(index)), chain_lam)
                    if strategy['tau'] == True:
                        np.save(self.path_tau +  'sample_tau_' + np.str(np.int(index)), chain_tau)
                    if strategy['u0'] == True:
                        np.save(self.path_u0 + 'sample_u0_' + np.str(np.int(index)), chain_u0)
                    del chain_u, chain_lam, chain_tau
                    chain_u, chain_lam, chain_tau, chain_u0 = [], [], [], []
                    index += 1

            i += 1

        return (chain_u, chain_u0, chain_lam, chain_tau), (ac_rate_u, ac_rate_u0, ac_rate_lam, ac_rate_tau)
        



























