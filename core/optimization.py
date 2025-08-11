#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:41:27 2019

@author: jjx323
"""
import numpy as np
import scipy.sparse.linalg as spsl

###########################################################################
class EvaMAP(object):
    def __init__(self, model, max_iter=500, tol=0.1/100):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.m_est, self.loss_all = None, None
        
    def evaluate_MAP(self, m0=None, method='gradient_descent', show_step=False, callback=None, option=None):
        if method is 'gradient_descent':
            return self.gradient_descent(m0, show_step, callback)
        elif method is 'newton_cg':
            return self.newton_cg(m0, show_step, callback, option)
        elif method is 'gd_prior_precondition':
            return self.gd_prior_precondition(m0, show_step, callback)
    
    def armijo_line_search(self, mk, g, cost_old, show_step=False):
        converged = True
        mk_pre = mk.copy()
        c_armijo = 1e-5
        step_length = 1
        backtrack_converged = False
        grad_norm2 = np.sqrt(g@self.model.M_coef@g)
        # grad_norm2 = np.max(np.abs(g))
        # grad_norm2 = np.max(np.power(g, 2))
        # grad_norm2 = np.max(np.power(g, 1))
        # print('grad_norm2 ', grad_norm2)
        for it_backtrack in range(20):
            mk = mk + step_length*(g)/max(grad_norm2+1e-15, 1)
            self.model.update_m(mk, update_sol=True)
            cost_all = self.model.loss()
            cost_new = cost_all[0]
#            print('cost_new ', cost_new)
            if cost_new < cost_old - step_length*c_armijo*grad_norm2:
                cost_old = cost_new
                backtrack_converged = True
                break
            else:
                step_length*=0.5
                mk = mk_pre.copy()
            if show_step == True:
                print("search num is ", it_backtrack, " step_length is ", step_length)
        if backtrack_converged == False:
            print("Backtracking failed. A sufficient descent direction was not found")
            converged = False
        return mk, cost_all, converged

    def error_change(self, mk_pre, mk):
        fenmu = mk_pre@self.model.M_coef@mk_pre
        fenzi = (mk_pre - mk)@self.model.M_coef@(mk_pre - mk)
        return fenzi/(fenmu+1e-15)
        
    def newton_cg(self, m0=None, show_step=False, callback=None, option=None):
        if m0 is None:
            m0 = np.zeros((len(self.model.m.vector()[:]),))
        if option == None:
            cg_tol, cg_maxiter =1e-5, 100
        else:
            cg_tol, cg_maxiter = option['cg_tol'], option['cg_maxiter'] 
        loss_all = []
        iter_num = 1
        diff = 1
        mk = m0.copy()
        mk_pre = m0.copy()
        self.model.update_m(mk, update_sol=True)
        cost_all = self.model.loss()
        cost = cost_all[0]
        loss_all.append(cost)
        #print(cost)
        tt = 1
        #stop_value = 1.96*np.sqrt((tt/tau))/np.sqrt(len(d))  # 95% confidence region
        #stop_value = 0.845*np.sqrt((tt/tau))/np.sqrt(len(d))  # 60% confidence region
        #stop_value = 0.255*np.sqrt((tt/tau))/np.sqrt(len(d)) # 20% confidence region
        #stop_value = 0.125*np.sqrt((tt/tau))/np.sqrt(len(d)) # 10% confidence region
#        stop_value = 0.065*np.sqrt((tt/self.model.tau))/np.sqrt(len(self.model.d)) # 5% confidence region
        stop_value = 0
        stop_mean, g1 = 1, 1
        while iter_num <= self.max_iter and diff > self.tol and np.linalg.norm(g1) > self.tol:# and stop_mean > stop_value:
            self.model.update_m(mk, update_sol=False)
            g_all = self.model.gradient(mk)
            g1 = g_all[0]
            hessian_operator = self.model.hessian_linear_operator()
            pre_cond = self.model.precondition_linear_operator()
#            def callback1(x):
#                #print("residual_p, ", np.linalg.norm(hessian_operator*gp+g1))
#                print("residual ", np.linalg.norm(hessian_operator*x+g1))
#            g, _ = spsl.cg(hessian_operator, -g1, M=pre_cond, tol=1e-3, \
#                                 maxiter=5, callback=callback1)
            ## ----------------------------------------------------------------
#            g, _ = spsl.cg(hessian_operator, -g1, M=pre_cond, tol=cg_tol, maxiter=cg_maxiter)
#            g, _ = spsl.cgs(hessian_operator, -g1, M=pre_cond, tol=cg_tol, maxiter=cg_maxiter)
#            g, _ = spsl.gmres(hessian_operator, -g1, M=pre_cond, tol=cg_tol, maxiter=cg_maxiter)
#            g, _ = spsl.lgmres(hessian_operator, -g1, M=pre_cond, tol=cg_tol, maxiter=cg_maxiter)
#            g, _ = spsl.minres(hessian_operator, -g1, M=pre_cond, tol=cg_tol, maxiter=cg_maxiter)           
            g, _ = spsl.bicgstab(hessian_operator, -g1, M=pre_cond, tol=cg_tol, maxiter=cg_maxiter)
            ## ----------------------------------------------------------------
#            g, _ = spsl.cg(hessian_operator, -g1, tol=cg_tol, maxiter=cg_maxiter)
            mk, cost_all, converged = self.armijo_line_search(mk, g, cost, show_step=show_step)
            cost, cost_res = cost_all[0], cost_all[1]
            if converged is False:
                info = 4
                break
            diff = self.error_change(mk_pre, mk)
            stop_mean = np.abs(np.mean(cost_res))
            mk_pre = mk.copy()
            loss_all.append(cost)
            if show_step is True:
                print("\033[0;31m%s\033[0m" % "iter_num is ", "\033[0;31m%s\033[0m" % iter_num, \
                      "\033[0;31m%s\033[0m" % " cost is ", "\033[0;31m%s\033[0m" % cost)
            iter_num += 1
            if callback is not None:
                callback(mk.copy())
#        self.model.update_m(mk, update_sol=False)
        ## end while
        self.loss_all = loss_all
        self.m_est = mk.copy()
        
        if iter_num > self.max_iter: 
            info = 1
        elif diff <= self.tol:
            info = 2
        elif stop_mean <= stop_value:
            info = 3
        else:
            info = 4
            
        return self.m_est, self.loss_all, info  
            
    def gradient_descent(self, m0=None, show_step=False, callback=None):
        if m0 is None:
            m0 = np.zeros((len(self.model.m.vector()[:]),))
        loss_all = []
        iter_num = 1
        diff = 1
        mk = m0.copy()
        mk_pre = m0.copy()
        self.model.update_m(mk, update_sol=True)
        cost_all = self.model.loss()
        cost = cost_all[0]
        loss_all.append(cost)
        #print(cost)
        tt = 1
        #stop_value = 1.96*np.sqrt((tt/tau))/np.sqrt(len(d))  # 95% confidence region
        #stop_value = 0.845*np.sqrt((tt/tau))/np.sqrt(len(d))  # 60% confidence region
        #stop_value = 0.255*np.sqrt((tt/tau))/np.sqrt(len(d)) # 20% confidence region
        #stop_value = 0.125*np.sqrt((tt/tau))/np.sqrt(len(d)) # 10% confidence region
        stop_value = 0.065*np.sqrt((tt/self.model.tau))/np.sqrt(len(self.model.d)) # 5% confidence region
        stop_mean = 1
        while iter_num <= self.max_iter and diff > self.tol:# and stop_mean > stop_value: 
            g_all = self.model.gradient(mk)
            g = g_all[0]
            mk, cost_all, converged = self.armijo_line_search(mk, -g, cost, show_step=show_step)
            cost, cost_res = cost_all[0], cost_all[1]
            if converged is False:
                info = 4
                break
            diff = self.error_change(mk_pre, mk)
            stop_mean = np.abs(np.mean(cost_res))
            mk_pre = mk.copy()
            loss_all.append(cost)
            if show_step is True:
                print("\033[0;31m%s\033[0m" % "iter_num is ", "\033[0;31m%s\033[0m" % iter_num, \
                      "\033[0;31m%s\033[0m" % " cost is ", "\033[0;31m%s\033[0m" % cost)
            iter_num += 1
            if callback is not None:
                callback(mk.copy())
        ## end while
        self.loss_all = loss_all
        self.m_est = mk.copy()
        
        if iter_num > self.max_iter: 
            info = 1
        elif diff <= self.tol:
            info = 2
        elif stop_mean <= stop_value:
            info = 3
        else:
            info = 4
            
        return self.m_est, self.loss_all, info
    
    def gd_prior_precondition(self, m0=None, show_step=False, callback=None):
        if m0 is None:
            m0 = np.zeros((len(self.model.m.vector()[:]),))
        loss_all = []
        iter_num = 1
        diff = 1
        mk = m0.copy()
        mk_pre = m0.copy()
        self.model.update_m(mk, update_sol=True)
        cost_all = self.model.loss()
        cost = cost_all[0]
        loss_all.append(cost)
        #print(cost)
        tt = 1
        #stop_value = 1.96*np.sqrt((tt/tau))/np.sqrt(len(d))  # 95% confidence region
        #stop_value = 0.845*np.sqrt((tt/tau))/np.sqrt(len(d))  # 60% confidence region
        #stop_value = 0.255*np.sqrt((tt/tau))/np.sqrt(len(d)) # 20% confidence region
        #stop_value = 0.125*np.sqrt((tt/tau))/np.sqrt(len(d)) # 10% confidence region
        stop_value = 0.065*np.sqrt((tt/self.model.tau))/np.sqrt(len(self.model.d)) # 5% confidence region
        stop_mean = 1
        while iter_num <= self.max_iter and diff > self.tol:# and stop_mean > stop_value: 
            g_all = self.model.gradient(mk)
            g = g_all[0]
            g_precondition = self.model.ref_u.precondition(g)
            mk, cost_all, converged = self.armijo_line_search(mk, -g_precondition, cost, show_step=show_step)
            cost, cost_res = cost_all[0], cost_all[1]
            if converged is False:
                info = 4
                break
            diff = self.error_change(mk_pre, mk)
            stop_mean = np.abs(np.mean(cost_res))
            mk_pre = mk.copy()
            loss_all.append(cost)
            if show_step is True:
                print("\033[0;31m%s\033[0m" % "iter_num is ", "\033[0;31m%s\033[0m" % iter_num, \
                      "\033[0;31m%s\033[0m" % " cost is ", "\033[0;31m%s\033[0m" % cost)
            iter_num += 1
            if callback is not None:
                callback(mk.copy())
        ## end while
        self.loss_all = loss_all
        self.m_est = mk.copy()
        
        if iter_num > self.max_iter: 
            info = 1
        elif diff <= self.tol:
            info = 2
        elif stop_mean <= stop_value:
            info = 3
        else:
            info = 4
            
        return self.m_est, self.loss_all, info













