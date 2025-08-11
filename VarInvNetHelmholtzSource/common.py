#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:23:53 2021

@author: jjx323
"""

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
import dolfin as dl 
from operator import mul
from functools import reduce
import torch

import sys, os
sys.path.append(os.pardir)
from core.probability import GaussianMeasure
from core.model import Model_Hybrid
from core.misc import trans_to_python_sparse_matrix, make_interpolation_matrix, \
                      MY_Project


class DomainPML(object):
    def __init__(self, nx=100, ny=100, dPML=0.1, xx=1.0, yy=1.0, sig0=1.5, p=2.3):
        self.nx, self.ny = nx, ny
        self.dPML = dPML
        self.sig0, self.p = sig0, p
        self.xx, self.yy = xx, yy
        dPML, xx, yy, Nx, Ny = self.dPML, self.xx, self.yy, self.nx, self.ny
        self.mesh = fe.RectangleMesh(fe.Point(-dPML, -dPML), fe.Point(xx+dPML, yy+dPML), Nx, Ny)
    
        P2 = fe.FiniteElement('P', fe.triangle, 1)
        self.V1 = fe.FunctionSpace(self.mesh, P2)
        element = fe.MixedElement([P2, P2])
        self.V = fe.FunctionSpace(self.mesh, element)
        self.VR, self.VI = self.V.sub(0).collapse(), self.V.sub(1).collapse()
        
    def modifyNxNy(self, nx, ny):
        dPML, xx, yy, Nx, Ny = self.dPML, self.xx, self.yy, self.nx, self.ny
        self.mesh = fe.RectangleMesh(fe.Point(-dPML, -dPML), fe.Point(xx+dPML, yy+dPML), Nx, Ny)
        
    def numberOfMesh(self):
        return self.nx*self.ny*2
    
    def sizeOfMesh(self):
        xlen, ylen = self.xx/self.nx, self.yy/self.ny
        return [xlen, ylen, np.sqrt(xlen**2 + ylen**2), 0.5*xlen*ylen]


class EquSolverPML(object):
    def __init__(self, domain, kappa=5, q_fun=fe.Constant(0.0), fR=fe.Constant(0.0),\
                 fI=fe.Constant(0.0), points=None):
        self.domain = domain
        self.V = self.domain.V
        self.kappa = kappa
        # define boundary conditions
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc = [fe.DirichletBC(self.V.sub(0), fe.Constant(0.0), boundary), \
                   fe.DirichletBC(self.V.sub(1), fe.Constant(0.0), boundary)]
        
        xx, yy, dPML, sig0_, p_ = self.domain.xx, self.domain.yy, self.domain.dPML,\
                                  self.domain.sig0, self.domain.p
        # define the coefficents induced by PML
        sig1 = fe.Expression('x[0] > x1 && x[0] < x1 + dd ? sig0*pow((x[0]-x1)/dd, p) : (x[0] < 0 && x[0] > -dd ? sig0*pow((-x[0])/dd, p) : 0)', 
                     degree=3, x1=xx, dd=dPML, sig0=sig0_, p=p_)
        sig2 = fe.Expression('x[1] > x2 && x[1] < x2 + dd ? sig0*pow((x[1]-x2)/dd, p) : (x[1] < 0 && x[1] > -dd ? sig0*pow((-x[1])/dd, p) : 0)', 
                     degree=3, x2=yy, dd=dPML, sig0=sig0_, p=p_)
        
        self.sR = fe.as_matrix([[(1+sig1*sig2)/(1+sig1*sig1), 0.0], [0.0, (1+sig1*sig2)/(1+sig2*sig2)]])
        self.sI = fe.as_matrix([[(sig2-sig1)/(1+sig1*sig1), 0.0], [0.0, (sig1-sig2)/(1+sig2*sig2)]])
        self.cR = 1 - sig1*sig2
        self.cI = sig1 + sig2
        
        self.VR, self.VI = self.domain.VR, self.domain.VI
        
        self.q_fun = fe.interpolate(q_fun, self.VR)
        self.fR = fe.interpolate(fR, self.VR)
        self.fI = fe.interpolate(fI, self.VI)
        
        self.points = points
        if type(self.points) != type(None):
            self.SR = make_interpolation_matrix(points, self.VR).toarray()
            self.SI = make_interpolation_matrix(points, self.VI).toarray()
            a, b = self.SR.shape
            self.S = np.zeros((2*a, 2*b))
            for i in range(a):
                sri = np.array(self.SR[i,:]).ravel()
                temp = np.zeros((b,)).ravel()
                si = np.array((np.row_stack((sri, temp)).T).reshape(np.int(2*b)))
                self.S[2*i,:] = si
                sii = np.array(self.SI[i,:]).ravel()
                s2i = np.array((np.row_stack((temp, sii)).T).reshape(np.int(2*b)))
                self.S[2*i+1,:] = s2i
            self.S = np.array(self.S)

        u_ = fe.TestFunction(self.V)
        v_ = fe.TrialFunction(self.V)
        M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.M = trans_to_python_sparse_matrix(M_)
        
    def geneSecondOrderMatrix(self):
        u_ = fe.TestFunction(self.V)
        du = fe.TrialFunction(self.V)
        u_R, u_I = fe.split(u_)
        duR, duI = fe.split(du)
        def sigR(v):
            return fe.dot(self.sR, fe.nabla_grad(v))
        def sigI(v):
            return fe.dot(self.sI, fe.nabla_grad(v))
        
        a1 = - fe.inner(sigR(duR)-sigI(duI), fe.nabla_grad(u_R))*(fe.dx) \
            - fe.inner(sigR(duI)+sigI(duR), fe.nabla_grad(u_I))*(fe.dx) 
            # - self.fR*u_R*(fe.dx) - self.fI*u_I*(fe.dx)
            
        self.A1 = fe.assemble(a1)
        self.bc[0].apply(self.A1)
        self.bc[1].apply(self.A1)
        
    def geneF(self):
        u_ = fe.TestFunction(self.V)
        du = fe.TrialFunction(self.V)
        u_R, u_I = fe.split(u_)
        duR, duI = fe.split(du)
        b1 = self.fR*u_R*(fe.dx) + self.fI*u_I*(fe.dx)
        self.b1 = fe.assemble(b1)
        self.bc[0].apply(self.b1)
        self.bc[1].apply(self.b1)
        
    def geneFirstOrderMatrix(self):
        u_ = fe.TestFunction(self.V)
        du = fe.TrialFunction(self.V)
        
        u_R, u_I = fe.split(u_)
        duR, duI = fe.split(du)
        
        q_fun = fe.interpolate(fe.Constant(1.0), self.q_fun.function_space()) + self.q_fun
        
        a2 = fe.inner(q_fun*(self.cR*duR-self.cI*duI), u_R)*(fe.dx) \
             + fe.inner(q_fun*(self.cR*duI+self.cI*duR), u_I)*(fe.dx) 
             
        self.A2 = fe.assemble(a2)
        self.bc[0].apply(self.A2)
        self.bc[1].apply(self.A2)
        
    def update_k(self, k):
        self.kappa = k
        
    def update_f(self, fR_vec):
        self.fR.vector()[:] = np.array(fR_vec)
        # self.fI.vector()[:] = np.array(f[1::2])
        self.geneF()
        # rhsR = self.fR.vector()[:]
        # rhsI = self.fI.vector()[:]
        # rhs = (np.row_stack((rhsR, rhsI)).T).reshape(np.int(2*len(rhsR)))
        # self.b1 = self.M@rhs
    
    def geneForwardNumpyMatrix(self): 
        self.geneSecondOrderMatrix()
        self.geneFirstOrderMatrix()
        self.geneF()
        self.A1ForwardNumpy = trans_to_python_sparse_matrix(self.A1)
        self.A2ForwardNumpy = trans_to_python_sparse_matrix(self.A2)
    
    def forward_solve(self):
        self.forward_sol_vec = spsl.spsolve(self.A1ForwardNumpy+\
                                self.kappa*self.kappa*self.A2ForwardNumpy, np.array(self.b1[:]))
        
    def geneAdjointSecondOrderMatrix(self):
        u_ = fe.TestFunction(self.V)
        du = fe.TrialFunction(self.V)
        u_R, u_I = fe.split(u_)
        duR, duI = fe.split(du)
        def sigR(v):
            return fe.dot(self.sR, fe.nabla_grad(v))
        def sigI(v):
            return fe.dot(self.sI, fe.nabla_grad(v))
        
        a1 = - fe.inner(sigR(duR)+sigI(duI), fe.nabla_grad(u_R))*(fe.dx) \
            - fe.inner(sigR(duI)-sigI(duR), fe.nabla_grad(u_I))*(fe.dx) 
            
        self.A1Adjoint = fe.assemble(a1)
        self.bc[0].apply(self.A1Adjoint)
        self.bc[1].apply(self.A1Adjoint)
        
    def geneAdjointFirstOrderMatrix(self):
        u_ = fe.TestFunction(self.V)
        du = fe.TrialFunction(self.V)
        
        u_R, u_I = fe.split(u_)
        duR, duI = fe.split(du)
        
        q_fun = fe.interpolate(fe.Constant(1.0), self.q_fun.function_space()) + self.q_fun
        
        a2 = fe.inner(q_fun*(self.cR*duR+self.cI*duI), u_R)*(fe.dx) \
             + fe.inner(q_fun*(self.cR*duI-self.cI*duR), u_I)*(fe.dx) 
             
        self.A2Adjoint = fe.assemble(a2)
        self.bc[0].apply(self.A2Adjoint)
        self.bc[1].apply(self.A2Adjoint)
        
    def geneAdjointNumpyMatrix(self): 
        self.geneAdjointSecondOrderMatrix()
        self.geneAdjointFirstOrderMatrix()
        self.A1AdjointNumpy = trans_to_python_sparse_matrix(self.A1Adjoint)
        self.A2AdjointNumpy = trans_to_python_sparse_matrix(self.A2Adjoint)
        
    def adjoint_solve(self, res_vec):
        # res_vecR, res_vecI = res_vec[::2], res_vec[1::2]
        # rhsR = self.SR.T@res_vecR 
        # rhsI = self.SI.T@res_vecI 
        # rhs = (np.row_stack((rhsR, rhsI)).T).reshape(np.int(2*len(rhsR)))
        rhs = self.S.T@np.array(res_vec)
        A = self.A1AdjointNumpy + self.kappa*self.kappa*self.A2AdjointNumpy
        self.adjoint_sol_vec = np.array(spsl.spsolve(A, rhs))
    
    def get_data(self, w):
        # wR, wI = np.arrary(self.w[::2]), np.array(self.w[1::2])
        # dataR = self.S1@w[::2]
        # dataI = self.S2@w[1::2]
        # data = (np.row_stack((dataR, dataI)).T).reshape(np.int(2*len(dataR)))
        # return data
        return np.array(self.S@w)

class EquSolverPMLSmall(EquSolverPML):
    def __init__(self, domain, kappa=5, q_fun=fe.Constant(0.0), fR=fe.Constant(0.0),\
                 fI=fe.Constant(0.0), points=None):
        super(EquSolverPMLSmall, self).__init__(domain, kappa, q_fun, fR, fI, points)
        self.geneForwardNumpyMatrix()
        self.geneAdjointNumpyMatrix()
        self.geneF()
        self.b1Numpy = np.array(self.b1[:])
               
    def to_dense(self):
        self.A1ForwardNumpy = np.array(self.A1ForwardNumpy.todense())
        self.A2ForwardNumpy = np.array(self.A2ForwardNumpy.todense())
        self.A1AdjointNumpy = np.array(self.A1AdjointNumpy.todense())
        self.A2AdjointNumpy = np.array(self.A2AdjointNumpy.todense())
        self.M = np.array(self.M.todense())
    
    def to_cuda(self):
        pass
    
    def eva_inv(self):
        k2 = self.kappa*self.kappa
        Ainv = np.linalg.inv(self.A1ForwardNumpy+k2*self.A2ForwardNumpy)
        Binv = np.linalg.inv(self.A1AdjointNumpy+k2*self.A2AdjointNumpy)
        return Ainv, Binv
        
    def update_f(self, fR_vec):
        self.fR.vector()[:] = np.array(fR_vec)
        # self.fI.vector()[:] = np.array(f[1::2])
        self.geneF()
        self.b1Numpy = np.array(self.b1[:])
        
    def forward_solve(self, Ainv=None):
        if type(Ainv) != type(None):
            return Ainv@self.b1Numpy
        else:
            k2 = self.kappa*self.kappa
            sol = spsl.spsolve(self.A1ForwardNumpy+k2*self.A2ForwardNumpy, self.b1Numpy)
            return np.array(sol)
          
    def adjoint_solve(self, res_vec, Binv=None):
        res_vecR, res_vecI = res_vec[::2], res_vec[1::2]
        rhsR = self.SR.T@res_vecR 
        rhsI = self.SI.T@res_vecI 
        rhs = (np.row_stack((rhsR, rhsI)).T).reshape(np.int(2*len(rhsR)))
        if type(Binv) != type(None):
            return Binv@rhs
        else:
            k2 = self.kappa*self.kappa
            sol = spsl.spsolve(self.A1AdjointNumpy+k2*self.A2AdjointNumpy, rhs)
            return np.array(sol)


class EquSolver(object):
    def __init__(self, domain, f, u, points, k=1, g=None):
        self.domain = domain
        self.V = self.domain.function_space
        self.f = fe.interpolate(f, self.V)
        if type(g) == type(None):
            g = fe.Constant("0.0")
        self.g = fe.interpolate(g, self.V)
        self.kappa = k
        self.kappa2 = k**2
        self.u = fe.interpolate(u, self.V)
        self.exp_u = fe.Function(self.V)
        self.my_project = MY_Project(self.V)
        self.exp_u.vector()[:] = self.my_project.project(dl.exp(self.u))
        self.S = make_interpolation_matrix(points, self.V)
        self.w = fe.TrialFunction(self.V) 
        self.v = fe.TestFunction(self.V) 
        self.K_ = fe.assemble(fe.inner(fe.grad(self.w), fe.grad(self.v))*fe.dx)
        self.K = trans_to_python_sparse_matrix(self.K_)
        self.M_ = fe.assemble(fe.inner(self.w, self.v)*fe.dx)
        self.M = trans_to_python_sparse_matrix(self.M_)
        self.FF_ = fe.assemble(-self.f*self.v*fe.dx + self.g*self.v*fe.ds)
        self.FF = self.FF_[:]
        self.B_ = fe.assemble(fe.inner(self.exp_u*self.w, self.v)*fe.dx)
        self.B = trans_to_python_sparse_matrix(self.B_)
        self.FG_ = fe.assemble(self.g*self.v*fe.ds)
        self.FG = self.FG_[:]
        self.F = self.FF[:] + self.FG[:]  
        
        self.sol_forward = fe.Function(self.V)
        self.sol_adjoint = fe.Function(self.V)
        self.sol_incremental = fe.Function(self.V)
        self.sol_incremental_adjoint = fe.Function(self.V)
        self.Fdu = fe.Function(self.V)
        
        self.Fs = fe.Function(self.V)
        self.temp_fun1 = fe.Function(self.V)
        self.temp_fun2 = fe.Function(self.V)
           
    def update_u(self, u_vec):
        self.u.vector()[:] = u_vec.copy()
        self.exp_u.vector()[:] = self.my_project.project(dl.exp(self.u)) 
        self.B_ = fe.assemble(fe.inner(self.exp_u*self.w, self.v)*fe.dx)
        self.B = trans_to_python_sparse_matrix(self.B_)
        
    def update_k(self, k):
        self.kappa = k
        self.kappa2 = k**2
        
    def update_f(self, f):
        self.f = fe.interpolate(f, self.V)
        self.FF_ = fe.assemble(-self.f*self.v*fe.dx)
        self.FF[:] = self.FF_[:]
        self.F = self.FF[:] + self.FG[:]
            
    def update_g(self, g):
        self.g = fe.interpolate(g, self.V)
        self.FG_ = fe.assemble(self.g*self.v*fe.ds)
        self.FG = self.FG_[:]
        self.F = self.FF[:] + self.FG[:]
        
    def get_data(self, points=None):
        if type(points) != type(None):
            self.S = make_interpolation_matrix(points, self.V)
        return self.S@self.sol_forward.vector()[:]

    def forward_solver(self):
        self.sol_forward.vector()[:] = np.array(spsl.spsolve(self.K - self.kappa2*self.B, self.F)) 
        # fe.solve(self.K_ - self.B_, self.sol_forward.vector(), self.F_)
        
    def adjoint_solver(self, d, noise):
        if type(noise.precision) != type(None):
            temp = noise.precision@(self.S@self.sol_forward.vector()[:] - noise.mean - d)
        else:
            temp = spsl.spsolve(noise.covariance, self.S@self.sol_forward.vector()[:] - \
                                noise.mean - d)
        F_adjoint = -self.S.T@temp 
        self.sol_adjoint.vector()[:] = np.array(\
                                    spsl.spsolve(self.K - self.kappa2*self.B, F_adjoint))       
        # self.Fs.vector()[:] = -self.S.T@temp
        # fe.solve(self.K_ - self.B_, self.sol_adjoint.vector(), self.Fs.vector())
    

class EquSolverS(EquSolver):
    def __init__(self, domain, f, u, points, noise, k, g=None):
        super(EquSolverS, self).__init__(domain, f, u, points, k, g=None)
        self.K = np.array(self.K.todense())
        self.B = np.array(self.B.todense())
        self.precision = noise.precision
        self.mean = noise.mean
        self.G = fe.assemble(self.g*self.v*fe.ds)[:]
        self.FM = trans_to_python_sparse_matrix(fe.assemble(-self.w*self.v*fe.dx)).todense()
        self.FM = np.array(self.FM)
    
    def to_cuda(self):
        pass
    
    def eva_inv(self):
        self.Ainv = np.linalg.inv(self.K - self.kappa2*self.B)
        
    def update_f(self, f):
        self.F = self.FM@f + self.G
        
    def forward_solver(self, Ainv=None):
        if type(Ainv) != type(None):
            self.Ainv = Ainv
        return self.Ainv@self.F
        # return np.linalg.solve(self.K - self.kappa2*self.B, self.F)
    
    def adjoint_solver(self, d, sol_forward, Ainv=None):
        if type(Ainv) != type(None):
            self.Ainv = Ainv
        temp = self.precision@(self.S@sol_forward - self.mean - d)
        F_adjoint = -self.S.T@temp
        return self.Ainv@F_adjoint
        # return np.linalg.solve(self.K - self.kappa2*self.B, F_adjoint)




























                      
                      
                      