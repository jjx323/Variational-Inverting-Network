#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:04:33 2019

@author: jjx323
"""
import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps

import fenics as fe 
import dolfin as dl

from core.misc import my_project, trans_to_python_sparse_matrix

#############################################################################
class Domain(object):
    def __init__(self, mesh_type='P', mesh_order=2):
        self.mesh_type = mesh_type
        self.mesh_order = mesh_order
        self._mesh = None
        self._function_space = None
        
    @property
    def mesh(self):
        return self._mesh
    
    @property
    def function_space(self):
        return self._function_space
    
    
class Domain2D(Domain):
    '''
    class Domain has to properties: mesh, function_space
    mesh: a square domain with uniform mesh
    function_space: can be specified by 'mesh_type' and 'mesh_order'
    '''
    def __init__(self, low_point=[0, 0], high_point=[1, 1], nx=100, ny=100, mesh_type='P', mesh_order=2):
        super().__init__(mesh_type, mesh_order)
        self._mesh = fe.RectangleMesh(fe.Point(low_point[0], low_point[1]), \
                                     fe.Point(high_point[0], high_point[1]), nx, ny)
        self._function_space = fe.FunctionSpace(self._mesh, self.mesh_type, self.mesh_order)
    
    def update(self, low_point=[0, 0], high_point=[1, 1], nx=100, ny=100, mesh_type='P', mesh_order=2):
        self.mesh_type = mesh_type
        self.mesh_order = mesh_order
        self._mesh = fe.RectangleMesh(fe.Point(low_point[0], low_point[1]), \
                                     fe.Point(high_point[0], high_point[1]), nx, ny)
        self._function_space = fe.FunctionSpace(self._mesh, mesh_type, mesh_order)
    

class Domain1D(Domain):
    def __init__(self, low_point=0, high_point=1, n=100, mesh_type='P', mesh_order=2):
        super().__init__(mesh_type, mesh_order)
        self._mesh = fe.IntervalMesh(n, low_point, high_point)
        self._function_space = fe.FunctionSpace(self._mesh, self.mesh_type, self.mesh_order)
        
    def update(self, low_point=0, high_point=1, n=100, mesh_type='P', mesh_order=2):
        self.mesh_type = mesh_type
        self.mesh_order = mesh_order
        self._mesh = fe.IntervalMesh(n, low_point, high_point)
        self._function_space = fe.FunctionSpace(self._mesh, mesh_type, mesh_order)
    
###########################################################################
class Model_Hybrid(object):
    def __init__(self, d, domain_coef, domain_equ, reference_u, R, lam, noise, equ_solver):
        self.domain_coef = domain_coef
        self.domain_equ = domain_equ
        self.ref_u = reference_u
        self.R = R
        self.lam = lam
        self.noise = noise
        self.tau = 1/max(noise.covariance.diagonal())
        self.equ_solver = equ_solver
        self.d = d
        self.p = fe.Function(self.domain_equ.function_space)
        self.q = fe.Function(self.domain_equ.function_space)
        self.m = fe.Function(self.domain_coef.function_space)
        self.grad_residual, self.grad_ref, self.grad_R = None, None, None
        self.hessian_residual, self.hessian_ref, self.hessian_R = None, None, None
        self.S = self.equ_solver.S
        u_ = fe.TrialFunction(self.domain_equ.function_space)
        v_ = fe.TestFunction(self.domain_equ.function_space)
        self.M_equ_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.M_equ = trans_to_python_sparse_matrix(self.M_equ_)
        u_ = fe.TrialFunction(self.domain_coef.function_space)
        v_ = fe.TestFunction(self.domain_coef.function_space)
        self.M_coef_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.M_coef = trans_to_python_sparse_matrix(self.M_coef_)
        temp_fun = fe.Function(domain_coef.function_space)
        self.geometric_dim = temp_fun.geometric_dimension()

    def update_m(self, m_vec, update_sol):
        pass
        
    def update_lam(self, lam):
        self.lam = lam

    def update_noise(self):
        pass
    
    def update_d(self, d):
        self.d = d.copy()
        
    def loss_residual(self):
        temp = (self.S@self.p.vector()[:] - self.d)
        temp = temp@temp
        return 0.5*temp
    
    def loss_R(self):
        if self.lam < 1e-15: 
            return 0.0
        else:
            return self.lam*self.R.evaluate_R(self.m.vector()[:])
    
    def loss_ref(self):
        return self.ref_u.evaluate_CM_norm(self.m.vector()[:])
    
    def loss(self):
        loss_residual = self.loss_residual()
        loss_R = self.loss_R()
        loss_ref = self.loss_ref()
        return loss_residual + loss_R + loss_ref, loss_residual, loss_R, loss_ref
        
    def eval_grad_residual(self, m_vec):
        Mgrad_residual = 0
        return Mgrad_residual
    
    def eval_grad_ref(self, m_vec):
        temp = self.ref_u.evaluate_grad(m_vec)
#        print(temp)
        return temp
        # return spsl.spsolve(self.M_coef, self.ref_u.evaluate_grad(m_vec))

    def eval_grad_R(self, m_vec):
        if self.lam < 1e-15: 
            return 0.0*m_vec
        else:
            return self.lam*self.R.evaluate_grad_R(m_vec)
            # return self.lam*spsl.spsolve(self.M_coef, self.R.evaluate_grad_R(m_vec))

    def gradient(self, m_vec):
        res = self.eval_grad_residual(m_vec)
        ref = self.eval_grad_ref(m_vec)
        r = self.eval_grad_R(m_vec)
        return res + r + ref, res, r, ref
        
    def eval_hessian_res_vec(self, m_hat_vec):
        HM = 0
        return HM
    
    def eval_hessian_r_vec(self, m_vec, m_current):
        if self.lam < 1e-15:
            return 0.0
        else:
            self.hessian_R = self.lam*self.R.evaluate_hessian_vec(m_vec, m_current)
            return self.hessian_R
    
    def eval_hessian_ref_vec(self, m_vec):
        self.hessian_ref = self.ref_u.evaluate_hessian_vec(m_vec)
        return self.hessian_ref
    
    def hessian(self, m_vec):
        res = self.eval_hessian_res_vec(m_vec)
        r = self.eval_hessian_r_vec(m_vec, self.m.vector()[:])
        ref = self.eval_hessian_ref_vec(m_vec)
        return res + r + ref
    
    # def hessian_r_ref(self, m_vec):
    #     # approximate the Hessian of residual term by Id
    #     res_ = self.tau*m_vec
    #     r = self.eval_hessian_r_vec(m_vec, self.m.vector()[:])
    #     ref = self.eval_hessian_ref_vec(m_vec)
    #     return res_ + r + ref
        
    def hessian_linear_operator(self):
        leng = self.M_coef.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self.hessian)
        return self.linear_ope
    
    # def hessian_linear_operator_r_ref(self):
    #     leng = self.M_coef.shape[0]
    #     self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self.hessian_r_ref)
    #     return self.linear_ope
    
    def precondition(self, m_vec):
        if self.lam < 1e-15:
            ## If no term R, we use the inverse of the covariance operator of reference measure
            ## as the preconditioner 
            return self.ref_u.precondition(m_vec)
        else:
            return spsl.spsolve(self.lam*self.R.grad_A, m_vec)
            # return spsl.spsolve(self.ref_u.K, self.ref_u.M@spsl.spsolve((self.ref_u.K).T, m_vec))

    def precondition_linear_operator(self):
        leng = self.M_coef.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self.precondition)
        return self.linear_ope
    































