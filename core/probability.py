#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:20:46 2019

@author: jjx323
"""
import numpy as np
import fenics as fe
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs, eigsh
import dolfin as dl 

from core.misc import trans_to_python_sparse_matrix, make_interpolation_matrix

#############################################################################
class GaussianMeasure(object):
    '''
    prior Gaussian probability measure N(m, C)
    C^{-1/2}: an elliptic operator -\alpha\nabla(\cdot\Theta\nabla \cdot) + \alpha_I Id
    '''
    def __init__(self, domain, alpha, theta, alpha_I = 1, mean_fun=None, tensor=False, boundary='Neumann', \
                 invM='full'):
        self.domain = domain
        self._alpha = alpha
        self._alpha_I = alpha_I
        self._tensor = tensor
        if tensor == False:
            self._theta = fe.interpolate(theta, self.domain.function_space)
        elif tensor == True:
            self._theta = fe.as_matrix(((fe.interpolate(theta[0], self.domain.function_space), \
                                         fe.interpolate(theta[1], self.domain.function_space)), \
                                        (fe.interpolate(theta[2], self.domain.function_space), \
                                         fe.interpolate(theta[3], self.domain.function_space))))
        if mean_fun == None:
            self.mean_fun = fe.interpolate(fe.Expression("0.0", degree=2), self.domain.function_space)
        else:
            self.mean_fun = fe.interpolate(mean_fun, self.domain.function_space)

        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        theta = self._theta
        a = fe.Constant(self._alpha)*fe.inner(theta*fe.grad(u), fe.grad(v))*fe.dx \
            + fe.Constant(self._alpha_I)*fe.inner(u, v)*fe.dx
        self.K_ = fe.assemble(a)
        b = fe.inner(u, v)*fe.dx
        self.M_ = fe.assemble(b)

        if boundary == 'Dirichlet':
            def boundary(x, on_boundary):
                return on_boundary
            bc = fe.DirichletBC(self.domain.function_space, fe.Constant('0.0'), boundary)
            bc.apply(self.K_)
            #bc.apply(self.M_)

        self.invM = invM
        self.K = trans_to_python_sparse_matrix(self.K_)
        self.M = trans_to_python_sparse_matrix(self.M_)
        self.Mid = sps.diags(1.0/self.M.diagonal())
        self.M_half = sps.diags(np.sqrt(self.M.diagonal()))
        self.M_half_ = fe.assemble(fe.inner(u, v)*fe.dx)
        self.Mid_ = fe.assemble(fe.inner(u, v)*fe.dx)
        self.M_half_.zero()
        v = fe.Vector()
        self.M_.init_vector(v, 1)
        v[:] = np.sqrt(self.M.diagonal())
        self.M_half_.set_diagonal(v)
        self.Mid_.zero()
        vv = fe.Vector()
        vv[:] = 1.0/self.M.diagonal()
        self.Mid_.set_diagonal(vv)

        ## auxillary functions
        self.temp0 = fe.Function(self.domain.function_space)
        self.temp1 = fe.Function(self.domain.function_space)
        self.temp2 = fe.Function(self.domain.function_space)
            
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, al):
        self._alpha = al
        self.generate_K()
    
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, th):
        if self._tensor == False:
            self._theta = th
        elif self._tensor == True:
            self._theta = fe.as_matrix(((fe.interpolate(th[0], self.domain.function_space), \
                                         fe.interpolate(th[1], self.domain.function_space)), \
                                        (fe.interpolate(th[2], self.domain.function_space), \
                                         fe.interpolate(th[3], self.domain.function_space))))
        self.generate_K()
        
    def update_mean_fun(self, mean_fun_vec):
        self.mean_fun.vector()[:] = mean_fun_vec
        
    def generate_K(self):
        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        theta = self._theta
        a = fe.Constant(self._alpha)*fe.inner(theta*fe.grad(u), fe.grad(v))*fe.dx \
            + fe.Constant(self._alpha_I)*fe.inner(u, v)*fe.dx
        self.K_ = fe.assemble(a)
        self.K = trans_to_python_sparse_matrix(self.K_)
        return self.K
    
    def generate_M(self):
        u = fe.TrialFunction(self.domain.function_space)
        v = fe.TestFunction(self.domain.function_space)
        a = fe.inner(u, v)*fe.dx
        self.M_ = fe.assemble(a)
        self.M = trans_to_python_sparse_matrix(self.M_)
        return self.M
    
    def generate_M_half(self, max_iter=2000, approTol=1e-4, method='diag_method'):
        if self.M == None:
            print("The matrix M has not been generated!")
            return None
        elif method == 'diag_method':
            self.M_half = sps.diags(np.sqrt(self.M.diagonal()))
            return self.M_half
        elif method == 'iter_method':
            nx, ny = self.M.shape
            alpha = 0.5
            Xk = (1.0/(2*alpha))*sps.eye(nx)
                
            for k in range(max_iter):
                Xk = Xk + alpha*(self.M - Xk.dot(Xk))                   
                ## set small elements to zeros to save memory
                row, col, vals = sps.find(Xk)
                index = (vals < approTol)
                rr, cc = row[index], col[index]
                Xk[rr, cc] = 0
        
            self.M_half = Xk
            return self.M_half
    
    def test_M_half(self):
        if self.M_half == None:
            print("The matrix M^{1/2} has not been generated!")
            return None
        else:
            error = []
    
            for i in range(500):
                e = np.random.normal(0, 1, (len(self.mean_fun.vector()[:]),))
                ee = self.M_half.dot(e)
                fenzi = np.abs(e.dot(self.M.dot(e)) - ee.dot(ee))
                fenmu = e.dot(self.M.dot(e))
                error.append(fenzi/fenmu)
    
            print("Average error is ", np.average(error), " and the maximum error is ", max(error))
            #plt.plot(error)
            return np.average(error)
    
    def prepare_for_generate_sample(self, approTol=1e-4, mode='diag'):
        if self.K == None:
            self.generate_K()
        if self.M_half == None:
            if mode == 'iter': 
                N = 2000
                self.generate_M_half(max_iter=N, approTol=approTol, method='iter_method')
                iter_num, max_num = 1, 3
                while self.test_M_half() > 0.01 and iter_num <= max_num:
                    N = N + 1000
                    self.generate_M_half(max_iter=N, approTol=approTol)
                    iter_num += 1
                if iter_num > max_num and self.test_M_half() > 0.01: 
                    print("Test of M_half is not pass!")
            elif mode == 'diag':
                _ = self.generate_M_half(method='diag_method')
            else:
                print("mode must be iter or diag!")
    
    def generate_sample(self, flag=None):
        '''
        generate samples from the Gaussian probability measure
        the generated vector is in $\mathbb{R}_{M}^{n}$ by
        $m = m_{mean} + Ln$ with $L:\mathbb{R}^{n}\rightarrow\mathbb{R}_{M}^{n}$
        '''
        if self.K == None:
            print("The matrix K has not been generated!")
            return None
        elif self.M_half == None:
            print("The matrix M^{1/2} has not been generated!")
            return None
        else:
            fun = fe.Function(self.domain.function_space)
            # n = np.random.normal(0, 1, (len(self.mean_fun.vector()[:]),))
            # fun.vector()[:] = self.mean_fun.vector()[:] + spsl.spsolve(self.K, self.M_half@n)
            n_ = fe.Vector()
            self.M_half_.init_vector(n_, 1)
            n_.set_local(np.random.normal(0, 1, (self.domain.function_space.dim(),)))
            # fe.solve(self.K_, fun.vector(), self.M_half_*n_, 'cg', 'ilu')
            fe.solve(self.K_, fun.vector(), self.M_half_*n_)
            fun.vector().axpy(1.0, self.mean_fun.vector())

            if flag == 'only_vec':
                return fun.vector()[:]
            elif flag == 'only_fun':
                return fun
            else:
                return fun, fun.vector()[:]
    
    def evaluate_CM_norm(self, u_vec):
        '''
        evaluate 0.5*\|u_vec\|_{CM}^{2}
        '''
        if type(u_vec) == np.ndarray:
            # temp = u_vec - self.mean_fun.vector()[:]
            # return 0.5*temp@(self.K.T)@spsl.spsolve(self.M, self.K@temp)
            self.temp0.vector()[:] = u_vec - self.mean_fun.vector()[:]
            if self.invM == 'full':
                fe.solve(self.M_, self.temp1.vector(), (self.K_*self.temp0.vector()))
            elif self.invM == 'simple':
                self.temp1.vector().set_local(self.Mid_*(self.K_*self.temp0.vector()))
            self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
            return 0.5*(self.temp0.vector()).inner(self.temp2.vector())
        else:
            print('Input value should be a numpy.array!')
            return None
        
    def evaluate_CM_inner(self, u_vec, v_vec):
        """
        evaluate (C^{-1/2}u, C^{-1/2}v)
        """
        if type(u_vec) == np.ndarray and type(v_vec) == np.ndarray:
            # temp1 = u_vec - self.mean_fun.vector()[:]
            # temp2 = v_vec - self.mean_fun.vector()[:]
            # return temp1@(self.K.T)@spsl.spsolve(self.M, self.K@temp2)
            self.temp0.vector()[:] = v_vec - self.mean_fun.vector()[:]
            if self.invM == 'full':
                fe.solve(self.M_, self.temp1.vector(), (self.K_*self.temp0.vector()))
            elif self.invM == 'simple':
                self.temp1.vector().set_local(self.Mid_*(self.K_*self.temp0.vector()))
            self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
            self.temp0.vector()[:] = u_vec - self.mean_fun.vector()[:]
            return (self.temp0.vector()).inner(self.temp2.vector())
        else:
            print("Input values should be a numpy.array!")
            return None
    
    def evaluate_grad(self, u_vec):
        '''
        calculate the gradient vector at u_vec
        the input vector should be in $\mathbb{R}_{M}^{n}$
        the output vector is in $v1\in\mathbb{R}_{M}^{n}$
        '''
        if type(u_vec) is np.ndarray:
        #     res = u_vec - self.mean_fun.vector()[:]
        #     grad_vec = (self.K.T)@spsl.spsolve(self.M, self.K@res)
        #     if flag == 'only_vec':
        #         return grad_vec
        #     elif flag == 'fun_vec':
        #         self.grad_ref = spsl.spsolve(self.M, grad_vec)
        #         return (self.grad_ref, grad_vec)
            if self.invM == 'full':
                self.temp0.vector()[:] = u_vec - self.mean_fun.vector()[:]
                fe.solve(self.M_, self.temp1.vector(), (self.K_*self.temp0.vector()))
                self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
                fe.solve(self.M_, self.temp1.vector(), self.temp2.vector())
            elif self.invM == 'simple':
                self.temp0.vector()[:] = u_vec - self.mean_fun.vector()[:]
                self.temp1.vector().set_local(self.Mid_*(self.K_*self.temp0.vector()))
                self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
                self.temp1.vector().set_local(self.Mid_*self.temp2.vector())
            return self.temp1.vector()[:]
        else:
            print('Input value should be a numpy.array!')
            return None
    
    def evaluate_hessian(self, u_vec):
        '''
        evaluate HessianMatrix^{-1}*(gradient at u_vec)
        '''
        return -u_vec
        
    def evaluate_hessian_vec(self, u_vec):
        '''
        evaluate HessianMatrix*u_vec
        the input vector should be in $\mathbb{R}_{M}^{n}$,
        the output vector is in $\mathbb{R}_{M}^{n}$
        '''
        if type(u_vec) is np.ndarray:
            # return (self.K.T)@spsl.spsolve(self.M, self.K@u_vec)
            if self.invM == 'full':
                self.temp0.vector()[:] = self.K@u_vec
                fe.solve(self.M_, self.temp1.vector(), self.temp0.vector())
                self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
                fe.solve(self.M_, self.temp1.vector(), self.temp2.vector())
            elif self.invM == 'simple':
                self.temp0.vector()[:] = self.K@u_vec
                self.temp1.vector().set_local(self.Mid_*self.temp0.vector())
                self.K_.transpmult(self.temp1.vector(), self.temp2.vector())
                self.temp1.vector().set_local(self.Mid_*self.temp2.vector())
            return self.temp1.vector()[:]
        else:
            print("Input should be a vector!")
            return None
        
    def precondition(self, m_vec):
        #return spsl.spsolve(self.K, self.M@spsl.spsolve((self.K).T, m_vec))
        return spsl.spsolve(self.K, self.M@spsl.spsolve((self.K).T, self.M@m_vec))
#        return m_vec

#############################################################################
class GaussianFractional(object):
    '''
    C = \alpha^{-1}(\delta_0 Id + \delta1 (-\nabla\cdot\theta\nable\cdot))^{-s}
    For 'dense=False', it has not been implemented and tested!
    '''
    def __init__(self, domain, alpha=1, s=2, delta1=1, delta0=1, mean_fun=None, \
                 theta=None, tensor=False, dense=False, boundary="Neumann", KL_trun=10):
        self.domain = domain
        self.alpha = alpha
        self.s = s
        self.delta1 = delta1
        self.delta0 = delta0
        self.KL_trun = KL_trun
        if theta == None:
            self._theta = fe.Function(self.domain.function_space)
            self._theta.vector()[:] = 1.0
        if tensor == False:
            if theta == None:
                self._theta = fe.interpolate(fe.Constant(1.0), self.domain.function_space)
            else:
                self._theta = fe.interpolate(theta, self.domain.function_space)
        elif tensor == True:
            self._theta = fe.as_matrix(((fe.interpolate(theta[0], self.domain.function_space), \
                                         fe.interpolate(theta[1], self.domain.function_space)), \
                                        (fe.interpolate(theta[2], self.domain.function_space), \
                                         fe.interpolate(theta[3], self.domain.function_space))))            
        self.boundary = boundary
        if mean_fun is None:
            self.mean_fun = fe.interpolate(fe.Constant("0.0"), self.domain.function_space)
        else:
            self.mean_fun = fe.interpolate(mean_fun, self.domain.function_space)
            
        u_, v_ = fe.TrialFunction(self.domain.function_space), fe.TestFunction(self.domain.function_space)
        self.M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.K_ = fe.assemble(fe.inner(self._theta*fe.grad(u_), fe.grad(v_))*fe.dx)
        self.Id = sps.eye(self.domain.function_space.dim())
        self.eigLam, self.eigV = None, None
        self.u = fe.Function(self.domain.function_space)
        self.dense = dense
        
        if self.boundary == "Dirichlet":
            def boundary(x, on_boundary):
                return on_boundary
            bc = fe.DirichletBC(self.domain.function_space, fe.Constant('0.0'), boundary)
            bc.apply(self.K_)
            bc.apply(self.M_)
            
        self.K = trans_to_python_sparse_matrix(self.K_)
        self.M = trans_to_python_sparse_matrix(self.M_)
        self.M_half = sps.diags(np.sqrt(self.M.diagonal()))
        self.rand_vec_len = 0
    
    def update_mean_fun(self, mean_fun_vec):
        self.mean_fun.vector()[:] = mean_fun_vec
    
    def prepare_for_generate_sample(self):
        if self.dense == False:
            '''
            Since eigsh function in scipy need the parameter k smaller than the matrix dimension, 
            so we need to reset the KL_trun when the specified KL_trun >= the matrix dimension. 
            '''
            if self.KL_trun >= self.domain.function_space.dim():
                self.KL_trun = self.domain.function_space.dim() - 1
                print("The KL expansion has been reset to %d terms" % self.KL_trun)
            '''
            Notice that the inverse of the eigenvalues of $M^{-1}K + I$ are the eigenvalues of the 
            covariance operator $\mathcal{C}$, we need to evaluate the smallest eigvalues and 
            corresponding eigenvectors that is the largest eigenvalues of the operator $\mathcal{C}$.
            '''
            eigLam2, eigV2 = spsl.eigsh(self.delta1*self.K+self.delta0*self.M, M=self.M, \
                                       k=self.KL_trun, which='SM')
            # eigLam2, eigV2 = spsl.eigs(self.delta1*self.K+self.delta0*self.M, M=self.M, \
            #                            k=self.KL_trun, which='SM')
            index = eigLam2 > 0  # only positive eigenvalues can take inverse
            self.eigLam = sps.diags(np.power(np.flip(np.real(eigLam2[index]), axis=0), -self.s/2))
            self.eigV = np.flip(eigV2[:, index], axis=1)
            Lam_2_inv = np.diag(np.power(self.eigLam.diagonal(), -2))
            Lam_2 = np.diag(np.power(self.eigLam.diagonal(), 2))
            
            self.c_inv = np.array(self.alpha*(self.eigV@Lam_2_inv@(self.eigV.T)@self.M))
            self.c = np.array((1/self.alpha)*(self.eigV@Lam_2@(self.eigV.T)@self.M))
            self.GM = (1/np.sqrt(self.alpha))*self.eigV@self.eigLam
            self.c_half = (1/np.sqrt(self.alpha))*self.eigV@self.eigLam@(self.eigV.T)@self.M
            self.rand_vec_len = self.eigLam.shape[0]
            self.precondition_c = np.array((1/self.alpha)*(self.eigV@Lam_2@(self.eigV.T)))
            self.eig_value_C = (1/self.alpha)*(Lam_2.diagonal())
        elif self.dense == True:
            '''
            When the self.dense == True, we can evaluate all eigenvalues. 
            (If we take the sparse mode, we can only evaluate self.domain.function_space.dim()-1
            number of eigenvalues)
            '''
            pass
            
    def reset_eign_value(self, eign_values):
        Lam_2 = np.diag(self.alpha*eign_values)
        Lam_2_inv = np.diag(np.power(self.alpha*eign_values, -1))
        self.eigLam = np.diag(np.power(self.alpha*eign_values, 0.5))
        
        self.c_inv = np.array(self.alpha*(self.eigV@Lam_2_inv@(self.eigV.T)@self.M))
        self.c = np.array((1/self.alpha)*(self.eigV@Lam_2@(self.eigV.T)@self.M))
        self.GM = (1/np.sqrt(self.alpha))*self.eigV@self.eigLam
        self.c_half = (1/np.sqrt(self.alpha))*self.eigV@self.eigLam@(self.eigV.T)@self.M
        self.rand_vec_len = self.eigLam.shape[0]
        self.precondition_c = np.array((1/self.alpha)*(self.eigV@Lam_2@(self.eigV.T)))
        self.eig_value_C = (1/self.alpha)*(Lam_2.diagonal())

    def generate_sample(self, flag='fun_vec'):
        a = np.random.normal(0, 1, (self.rand_vec_len,))
        vec = np.array(self.GM@a) + self.mean_fun.vector()[:]
        if flag == 'only_vec':
            return vec.flatten()
        else:
            fun = fe.Function(self.domain.function_space)
            fun.vector()[:] = vec.flatten()
            return fun, fun.vector()[:]
    
    def evaluate_CM_norm(self, u_vec):
        temp = u_vec - self.mean_fun.vector()[:]
        val = 0.5*temp@self.M@self.c_inv@temp
        return np.double(val)
    
    def evaluate_CM_inner(self, u_vec, v_vec):
        val = u_vec@self.M@self.c_inv@v_vec
        return np.double(val)
    
    def evaluate_grad(self, u_vec):
        '''
        calculate the gradient vector at u_vec
        '''
        if type(u_vec) == np.ndarray:
            temp = u_vec.flatten() - self.mean_fun.vector()[:]
            self.grad = self.c_inv@temp
#            return self.M@self.grad
            return self.grad
        else:
            print('Input value should be a numpy.array!')
            return None
    
    def evaluate_hessian_vec(self, u_vec, u_current=None):
        '''
        evaluate HessianMatrix*u_vec
        '''
        if type(u_vec) == np.ndarray:
#            return self.M@self.c_inv@u_vec
            return self.c_inv@u_vec
        else:
            print("Input should be a vector!")
            return None 
        
    def precondition(self, m_vec):
        return self.precondition_c@m_vec
#        return m_vec
        
#########################################################################
class Hybrid(object):
    def __init__(self, domain, mean_fun=fe.Constant('0.0'), boundary='Neumann',\
                 method='GradL1', ep=0.0001, invM='full'):
        self.domain = domain
        self.invM = invM  # it can be 'simple' or 'full'
        V = self.domain.function_space
        self.method = method
        self.u_, self.v_ = fe.TrialFunction(V), fe.TestFunction(V)
        self.A_ = fe.assemble(fe.inner(fe.grad(self.u_), fe.grad(self.v_))*fe.dx \
                             + fe.Constant('0.001')*fe.inner(self.u_, self.v_)*fe.dx)
        if boundary == 'Dirichlet':
            def boundary(x, on_boundary):
                return on_boundary
            bc = fe.DirichletBC(V, fe.Constant('0.0'), boundary)
            bc.apply(self.A_)
            
        self.grad_A = trans_to_python_sparse_matrix(self.A_)
        self.M_ = fe.assemble(fe.inner(self.u_, self.v_)*fe.dx)
        self.M = trans_to_python_sparse_matrix(self.M_)
        self.Mid = sps.diags(1.0/self.M.diagonal())

        self.M_half = sps.diags(np.sqrt(self.M.diagonal()))
        self.M_half_ = fe.assemble(fe.inner(self.u_, self.v_)*fe.dx)
        self.M_half_.zero()
        v = fe.Vector()
        self.M_.init_vector(v, 1)
        v[:] = np.sqrt(self.M.diagonal())
        self.M_half_.set_diagonal(v)

        self.fun = fe.Function(V)
        self.length = self.domain.function_space.dim()
        self.mean_fun = fe.interpolate(mean_fun, V)
        self.mean_vec = self.mean_fun.vector()[:]
        self.ep = fe.Constant(ep)
        self.grad_A4 = None
    
    def update_mean(self, mean_fun):
        if type(mean_fun) is np.ndarray:
            self.mean_fun.vector()[:] = mean_fun.copy()
            self.mean_vec = self.mean_fun.vector()[:]
        else:
            self.mean_fun.vector()[:] = mean_fun.vector()[:].copy()
            self.mean_vec = self.mean_fun.vector()[:]
        
    def evaluate_R(self, u):
        if self.method == 'GradL2':
            return self.evaluate_R_GradL2(u-self.mean_vec)
        elif self.method == 'GradL1':
            return self.evaluate_R_GradL1(u-self.mean_vec)
        elif self.method == 'Grad4L2':
            return self.evaluate_R_Grad4L2(u-self.mean_vec)
        elif self.method == 'Grad4L1':
            return self.evaluate_R_Grad4L1(u-self.mean_vec)
        
    def evaluate_R_GradL2(self, u):
        if self.length == len(u):
            return 0.5*u@self.grad_A@u
        else:
            print("Invalid inputs!")
    
    def evaluate_R_Grad4L2(self, u):
        if self.length == len(u):
            if self.invM == 'simple':
                return 0.5*u@(self.grad_A.T)@self.Mid@self.grad_A@u
            elif self.invM == 'full':
                return 0.5*u@(self.grad_A.T)@spsl.spsolve(self.M, self.grad_A@u)
        else:
            print("Invalid inputs!")
            
    def evaluate_R_GradL1(self, u):
        if self.length == len(u):
            uf = fe.Function(self.domain.function_space)
            uf.vector()[:] = u
            return fe.assemble(fe.sqrt(fe.inner(fe.grad(uf), fe.grad(uf)) + self.ep)*fe.dx)
        else:
            print("Invalid inputs!")

    def evaluate_R_Grad4L1(self, u):
        if self.length == len(u):
            uf = fe.Function(self.domain.function_space)
            uf.vector()[:] = u
            return fe.assemble(fe.sqrt(fe.inner(fe.div(fe.grad(uf)), fe.div(fe.grad(uf))) + self.ep)*fe.dx)
        else:
            print("Invalid inputs!")
            
    def evaluate_grad_R(self, u):
        if self.method == 'GradL2':
            return self.evaluate_grad_R_GradL2(u)
        elif self.method == 'GradL1':
            return self.evaluate_grad_R_GradL1(u)
        elif self.method == 'Grad4L2':
            return self.evaluate_grad_R_Grad4L2(u)
        elif self.method == 'Grad4L1':
            return self.evaluate_grad_R_Grad4L1(u)
            
    def evaluate_grad_R_GradL2(self, u):
        Mgrad = self.grad_A@u
        if self.invM == 'full':
            return spsl.spsolve(self.M, Mgrad)
        elif self.invM == 'simple':
            return self.Mid@Mgrad
        # return Mgrad
    
    def evaluate_grad_R_Grad4L2(self, u):
        if self.invM == 'full':
            Mgrad = (self.grad_A.T)@spsl.spsolve(self.M, self.grad_A@u)
            return spsl.spsolve(self.M, Mgrad)
        elif self.invM == 'simple':
            return self.Mid@(self.grad_A.T)@self.Mid@self.grad_A@u
        # return Mgrad

    def evaluate_grad_R_GradL1(self, u):
        uf = fe.Function(self.domain.function_space)
        uf.vector()[:] = u
        u_ = fe.TestFunction(self.domain.function_space)
        Mgrad = fe.assemble((1/fe.sqrt(fe.inner(fe.grad(uf), fe.grad(uf))+self.ep)*\
                     (fe.inner(fe.grad(uf), fe.grad(u_))))*fe.dx)[:]
        if self.invM == 'full':
            return spsl.spsolve(self.M, Mgrad)
        elif self.invM == 'simple':
            return self.Mid@Mgrad
        # return Mgrad

    def evaluate_grad_R_Grad4L1(self, u):
        uf = fe.Function(self.domain.function_space)
        uf.vector()[:] = u
        u_ = fe.TestFunction(self.domain.function_space)
        Mgrad = fe.assemble((1/fe.sqrt(fe.inner(fe.div(fe.grad(uf)), fe.div(fe.grad(uf)))+self.ep)*\
                     (fe.inner(fe.div(fe.grad(uf)), fe.div(fe.grad(u_)))))*fe.dx)[:]
        if self.invM == 'full':
            return spsl.spsolve(self.M, Mgrad)
        elif self.invM == 'simple':
            return self.Mid@Mgrad
        # return Mgrad

    def evaluate_hessian_vec(self, du, u=None):
        if self.method == 'GradL2':
            return self.evaluate_hessian_vec_GradL2(du)
        elif self.method == 'GradL1':
            return self.evaluate_hessian_vec_GradL1(du, u)
        elif self.method == 'Grad4L2':
            return self.evaluate_hessian_vec_Grad4L2(du)
        elif self.method == 'Grad4L1':
            return self.evaluate_hessian_vec_Grad4L1(du, u)
    
    def evaluate_hessian_vec_GradL2(self, du):
        # return self.grad_A@du
        if self.invM == 'full':
            return spsl.spsolve(self.M, self.grad_A@du)
        elif self.invM == 'simple':
            return self.Mid@self.grad_A@du

    def evaluate_hessian_vec_Grad4L2(self, du):
        # return (self.grad_A.T)@spsl.spsolve(self.M, self.grad_A@du)
        if self.invM == 'full':
            return spsl.spsolve(self.M, (self.grad_A.T)@spsl.spsolve(self.M, self.grad_A@du))
        elif self.invM == 'simple':
            return self.Mid@(self.grad_A.T)@self.Mid@self.grad_A@du
    
    def evaluate_hessian_vec_GradL1(self, du, u):
        uf = fe.Function(self.domain.function_space)
        uf.vector()[:] = u
        duu = fe.Function(self.domain.function_space)
        duu.vector()[:] = du
        u_ = fe.TestFunction(self.domain.function_space)
        term1 = fe.assemble((1/(fe.sqrt(fe.inner(fe.grad(uf), fe.grad(uf))+self.ep))\
                             *fe.inner(fe.grad(u_), fe.grad(duu)))*fe.dx)[:]
        term2 = fe.assemble((1/(fe.sqrt(fe.inner(fe.grad(uf), fe.grad(uf))+self.ep)*\
                               (fe.inner(fe.grad(uf), fe.grad(uf))+self.ep))*\
                            fe.inner(fe.grad(uf), fe.grad(u_))*\
                            fe.inner(fe.grad(uf), fe.grad(duu)))*fe.dx)[:]
        # return term1 - term2
        if self.invM == 'full':
            return spsl.spsolve(self.M, term1 - term2)
        elif self.invM == 'simple':
            return self.Mid@(term1 - term2)

    def evaluate_hessian_vec_Grad4L1(self, du, u):
        uf = fe.Function(self.domain.function_space)
        uf.vector()[:] = u
        duu = fe.Function(self.domain.function_space)
        duu.vector()[:] = du
        u_ = fe.TestFunction(self.domain.function_space)
        term1 = fe.assemble((1/(fe.sqrt(fe.inner(fe.div(fe.grad(uf)), fe.div(fe.grad(uf)))+self.ep))\
                             *fe.inner(fe.div(fe.grad(u_)), fe.div(fe.grad(duu))))*fe.dx)[:]
        term2 = fe.assemble((1/(fe.sqrt(fe.inner(fe.div(fe.grad(uf)), fe.div(fe.grad(uf)))+self.ep)*\
                               (fe.inner(fe.div(fe.grad(uf)), fe.div(fe.grad(uf)))+self.ep))*\
                            fe.inner(fe.div(fe.grad(uf)), fe.div(fe.grad(u_)))*\
                            fe.inner(fe.div(fe.grad(uf)), fe.div(fe.grad(duu))))*fe.dx)[:]
        # return term1 - term2
        if self.invM == 'full':
            return spsl.spsolve(self.M, term1 - term2)
        elif self.invM == 'simple':
            return self.Mid@(term1 - term2)

############################################################################
class GammaProbability(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.mean_value = self.alpha/self.beta
        
    def generate_sample(self, num=1):
        n = np.random.gamma(shape=self.alpha, scale=1/self.beta, size=num)
        return n
    
    def evaluate_mean_value(self):
        self.mean_value = self.alpha/self.beta
        return self.mean_value
    
    
###########################################################################
class PosteriorSamples(object):
    '''
    Using a group of samples to calculate the posterior information
    '''
    def __init__(self, V, samples):
        ## sample_list is a list in Python
        self.V = V  # V is the function space of samples
        self.samples_vec = samples
        self.length = len(samples)
        self.mean_vec = np.mean(np.array(self.samples_vec), axis=0)
        self.cov_fun = None
        self.mean_fun = fe.Function(self.V)
        self.mean_fun.vector()[:] = self.mean_vec.copy()

    def eval_mean(self, x_vec):
        S = make_interpolation_matrix(x_vec, self.V)
        return S@self.mean_fun.vector()[:]

    def eval_std(self, x_vec):
        std = 0
        S = make_interpolation_matrix(x_vec, self.V)
        for i in range(self.length):
            temp = S@self.samples_vec[i] - S@self.mean_fun.vector()[:]
            std = std + temp*temp
        std = std/self.length
        return std

    def eval_cov(self, x_vec, y_vec):
        cov = 0
        Sx = make_interpolation_matrix(x_vec, self.V)
        Sy = make_interpolation_matrix(y_vec, self.V)
        for i in range(self.length):
            temp_x = Sx@self.samples_vec[i] - Sx@self.mean_fun.vector()[:]
            temp_y = Sy@self.samples_vec[i] - Sy@self.mean_fun.vector()[:]
            cov = cov + temp_x*temp_y
        cov = cov/self.length
        return cov

############################################################################
class Noise(object):
    def __init__(self, dim):
        self.dim = dim
        self.mean = None
        self.covariance = None
        self.precision = None

    def set_parameters_Gaussian(self, mean_val=None, covariance_val=None, eval_precision=False):
        if type(mean_val) == type(None):
            # defalut value is zero
            self.mean = np.zeros(self.dim,)
        else:
            self.mean = mean_val

        if type(covariance_val) == type(None):
            # default value is identity matrix
            self.covariance = sps.eye(self.dim)
        else:
            self.covariance = sps.eye(self.dim)*covariance_val

        if eval_precision is True:
            self.precision = spsl.inv(self.covariance)

    def eval_precision(self):
        self.precision = spsl.inv(self.covariance)

    def update_paramters(self, mean_val=None, covariance_val=None):
        if type(mean_val) != type(None):
            self.mean = mean_val
        if type(covariance_val) != type(None):
            self.covariance = sps.eye(self.dim)*covariance_val
        return self.mean, self.covariance
    
    def generate_samples(self):
        a = np.random.normal(0, 1, (self.dim,))
        B = sps.diags(np.sqrt(self.covariance.diagonal()))
        vec = np.array(B@a)
        return vec








