import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
import scipy.linalg as sl
from scipy import sparse
import fenics as fe
import torch
import torch.nn.functional as F

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
from core.misc import trans_to_python_sparse_matrix
from core.model import Domain2D
from VarInvNetHelmholtzSource.common import DomainPML, EquSolverPML, EquSolverPMLSmall
        

def GenerateMf(V):
    u_, v_ = fe.TrialFunction(V), fe.TestFunction(V)
    M = fe.assemble(fe.inner(u_, v_)*fe.dx)
    M = trans_to_python_sparse_matrix(M)
    M_diag = M.diagonal()
    Minvhalf = sps.diags(np.power(M.diagonal(), -0.5))
    fun = fe.Function(V)
    # vec2matrix = gene_vec2matrix(V, device='cpu')
    # matrix2vec = gene_matrix2vec(V, device='cpu')
    def geneMf(nn):
        Mf_list = []
        num, chn, ln, rn = nn.shape
        for ii in range(num):
            for jj in range(chn):
                temp = np.array(nn[ii, jj, :]).ravel()
                ff = Minvhalf@temp
                fun.vector()[:] = ff
                Mf = fe.assemble(fe.inner(u_, fun*v_)*fe.dx)
                Mf_list.append(trans_to_python_sparse_matrix(Mf))
        return Mf_list, M, M_diag, Minvhalf, V
    return geneMf


def splitRI(z):
    return z[::2, :], z[1::2, :]


def combineRINumpy(aR, aI=None):
    if type(aI) == type(None):
        aI = aR*0.0
    if aR.ndim == 2:
        h, l = aR.shape 
        a = np.zeros((2*h, l))
        for i in range(h):
            a[2*i, :] = aR[i, :]
            a[2*i+1, :] = aI[i, :]
    elif aR.ndim == 1:
        h = aR.shape[0]
        a = (np.row_stack((aR, aI)).T).reshape(np.int(2*h))
    return np.array(a)

def combineRITorch(aR, aI=None):
    if type(aI) == type(None):
        aI = aR*0.0
    if aR.dim() == 2:
        h, l = aR.shape[0], aR.shape[1]
        a = torch.zeros((2*h, l))
        for i in range(h):
            a[2*i, :] = aR[i, :]
            a[2*i+1, :] = aI[i, :]
    elif aR.dim() == 1:
        h = aR.shape[0]
        a = (torch.row_stack((aR, aI)).T).reshape(2*h)
    return a


class ForwardOP(object):
    def __init__(self, mesh_size, f, coordinates, kappas=np.array([1.0])):
        self.mesh_size = mesh_size
        self.mesh = fe.UnitSquareMesh(mesh_size-1, mesh_size-1)
        self.ll = mesh_size*mesh_size
        self.Id_matrix = sparse.coo_matrix(sparse.eye(self.ll)) 
        equ_nx, equ_ny = mesh_size-1, mesh_size-1
        self.domainPML = DomainPML(nx=equ_nx, ny=equ_ny)
        self.domainS = Domain2D(nx=equ_nx, ny=equ_ny, mesh_type='CG', mesh_order=1)
        self.V = self.domainPML.VR
        self.kappas = kappas
        self.coordinates = coordinates
        fR = fe.interpolate(f, self.V)
        self.num_kappas = len(kappas)
        
        self.equ_solver = EquSolverPML(domain=self.domainPML, fR=fR, points=coordinates, \
                                       kappa=kappas[0])
        self.equ_solver.geneForwardNumpyMatrix()
        self.equ_solver.geneAdjointNumpyMatrix()
        self.M = sps.csr_matrix(self.equ_solver.M)
        
        # self.equ_solvers = []
        # for ii in range(self.num_kappas):
        #     self.equ_solvers.append(EquSolver(domain=self.domain, f=f, g=g, u=u, \
        #                                       points=coordinates, k=kappas[ii]))
        #     #print(ii)
        # self.M = sps.csr_matrix(self.equ_solvers[0].M)
        self.forward_matrixs = []
        for ii in range(self.num_kappas):
            # self.forward_matrixs.append(\
            #     sps.csr_matrix(self.equ_solvers[ii].K - \
            #                    self.equ_solvers[ii].kappa2*self.equ_solvers[ii].B))
            self.forward_matrixs.append(\
                sps.csr_matrix(self.equ_solver.A1ForwardNumpy + \
                     self.kappas[ii]*self.kappas[ii]*self.equ_solver.A2ForwardNumpy))
        
        self.forward_adjoint_matrixs = []
        for ii in range(self.num_kappas):
            self.forward_adjoint_matrixs.append(\
                sps.csr_matrix(self.equ_solver.A1AdjointNumpy + \
                     self.kappas[ii]*self.kappas[ii]*self.equ_solver.A2AdjointNumpy))
            
        # self.SR, self.SI = self.equ_solver.S1, self.equ_solver.S2
        self.S = self.equ_solver.S
        
        # self.len_dataR, self.len_fun = self.SR.shape
        # self.len_dataI, _ = self.SI.shape
        self.len_data, self.len_fun = self.S.shape
    
        self.matrix2vec_gpu = gene_matrix2vec(self.V, device='gpu')
        self.matrix2vec = gene_matrix2vec(self.V)
        self.vec2matrix_gpu = gene_vec2matrix(self.V, device='gpu')
        self.vec2matrix = gene_vec2matrix(self.V)        
        
    def forward_op(self, z, index_random=None):
        batch_size, ch, _, _ = z.shape
        if z.is_cuda:
            z_np = self.matrix2vec(z.detach().cpu().numpy())
            # z_np = self.matrix2vec_gpu(z).detach().cpu().numpu()
        else:
            z_np = self.matrix2vec(z.detach().numpy())

        zi_np = z_np.reshape(-1, self.ll).transpose((1, 0))
        
        if type(index_random) == type(None):
            kappas_all = list(np.arange(self.num_kappas))
        else:
            kappas_all = np.array(list(np.arange(self.num_kappas)))
            kappas_all = kappas_all[index_random]
        
        out = np.zeros((batch_size, ch, len(kappas_all), self.len_data))
        
        jj = 0
        zi_npRI = combineRINumpy(zi_np)
        for ii in kappas_all:
            temp = spsl.spsolve(self.forward_matrixs[ii], self.M@zi_npRI)
            # tempR, tempI = splitRI(temp)
            # out_tempR = (self.SR@tempR) #.reshape(self.len_dataR, -1)
            # out_tempI = (self.SI@tempI) #.reshape(self.len_dataI, -1)
            # out_temp = combineRINumpy(out_tempR, out_tempI)
            # out_temp = out_temp.reshape(self.len_dataR+self.len_dataI, -1)
            out_temp = self.S@temp
            out_temp = out_temp.reshape(self.len_data, -1)
            out_temp = out_temp.transpose((1,0))
            out[:,:,jj,:] = out_temp.reshape(-1, ch, self.len_data)
            jj += 1
            
        out_final = torch.from_numpy(out).to(device=z.device).type(dtype=z.dtype)
    
        return out_final
            
    
class ForwardOP_Rough(ForwardOP):
    def __init__(self, mesh_size, f, coordinates, kappas=np.array([1.0])):
        super().__init__(mesh_size, f, coordinates, kappas)
        for ii in range(self.num_kappas):
            if sps.isspmatrix(self.forward_matrixs[ii]) == True:
                self.forward_matrixs[ii] = self.forward_matrixs[ii].toarray()
            if sps.isspmatrix(self.forward_adjoint_matrixs[ii]) == True:
                self.forward_adjoint_matrixs[ii] = self.forward_adjoint_matrixs[ii].toarray()
            
        # self.precision = noise.precision.toarray()
        # self.mean = noise.mean

        self.M = self.M.toarray()
        temp, _ = self.M.shape
        self.sa = np.int(np.sqrt(temp))
        self.cuda = False
    
    def eva_inv(self, data_dir=''):
        self.Ainv = []
        for ii in range(self.num_kappas):
            self.Ainv.append(np.linalg.inv(self.forward_matrixs[ii]))
        self.Ainv = np.array(self.Ainv)
        np.save(data_dir + '/Ainvs', self.Ainv)
        
        self.Ainv_adjoint = []
        for ii in range(self.num_kappas):
            self.Ainv_adjoint.append(np.linalg.inv(self.forward_adjoint_matrixs[ii]))
        self.Ainv_adjoint = np.array(self.Ainv_adjoint)
        np.save(data_dir + '/Ainvs_adjoint', self.Ainv_adjoint)
        
    def load_inv(self, data_dir=''):
        self.Ainv = np.load(data_dir + '/Ainvs.npy')
        self.Ainv_adjoint = np.load(data_dir + '/Ainvs_adjoint.npy')
        
    def update_f(self, fR):
        f = combineRINumpy(fR)
        self.F = self.M@f
    
    def update_fTorch(self, fR):
        f = combineRITorch(fR)
        self.F = self.M@f
        
    # def forward_solver(self):
    #     data = np.zeros((self.num_kappas, self.len_data))
    #     for ii in range(self.num_kappas):
    #         data[ii,:] = self.S_forward@(self.Ainv[ii]@self.F)
        
    #     return data
    
    # def adjoint_solver(self, d, sol_forward, Ainv=None):
    #     if type(Ainv) != type(None):
    #         self.Ainv = Ainv
    #     temp = self.precision@(self.S@sol_forward - self.mean - d)
    #     F_adjoint = -self.S.T@temp
    #     return self.Ainv@F_adjoint
    #     # return np.linalg.solve(self.K - self.kappa2*self.B, F_adjoint)
        
    def inverting(self, data, init_val=None, step_length=torch.tensor(0.05, dtype=torch.float32)):        
        num_batch, num_ch, num_freq, num_data = data.shape
        len_funR = np.int(self.len_fun/2)
        f_iter_all = np.zeros((num_batch, num_freq, len_funR))
        f_iter_all = torch.tensor(f_iter_all, dtype=torch.float32)
        
        if type(init_val) != type(None):
            init_val = F.interpolate(init_val, size=[self.sa, self.sa], mode='bicubic')
            init_val = torch.tensor(self.matrix2vec_gpu(init_val), dtype=torch.float32)
        for jj in range(num_batch):
            if type(init_val) == type(None):
                f_iter = torch.tensor(np.zeros((len_funR, )), dtype=torch.float32)
            else:
                f_iter= init_val[jj, 0, :]
            for ii in range(self.num_kappas):
                for mm in range(1):
                    f_iterRI = combineRITorch(f_iter)
                    self.F = torch.matmul(self.M, f_iterRI) 
                    sol_forward = torch.matmul(self.Ainv[ii], self.F)
                    data_simulate = self.S@sol_forward
                    res_vec = data_simulate - data[jj, 0, ii,:]                    
                    F_adjoint = torch.matmul(self.S.T, res_vec)
                    sol_adjoint = torch.matmul(self.Ainv_adjoint[ii], F_adjoint)
                    grad_val = sol_adjoint[::2]
                    gnorm = torch.max(torch.abs(grad_val)) + 1e-15
                    f_iter += -step_length*grad_val/gnorm
                f_iter_all[jj, ii, :] = f_iter
        
        return torch.tensor(self.vec2matrix(f_iter_all), dtype=torch.float32)
    
    def to_tensor(self):
        self.Ainv = torch.tensor(self.Ainv, dtype=torch.float32)
        self.Ainv_adjoint = torch.tensor(self.Ainv_adjoint, dtype=torch.float32)
        self.M = torch.tensor(self.M, dtype=torch.float32)
        self.S = torch.tensor(self.S, dtype=torch.float32)

    def to_cuda(self):
        self.Ainv = torch.tensor(self.Ainv, dtype=torch.float32).cuda()
        self.Ainv_adjoint = torch.tensor(self.Ainv_adjoint, dtype=torch.float32).cuda()
        self.M = torch.tensor(self.M, dtype=torch.float32).cuda()
        self.S = torch.tensor(self.S, dtype=torch.float32).cuda()
        self.cuda = True
        
    # def inverting_cuda(self, data, init_val=None, step_length=torch.tensor(0.05, dtype=torch.float32)):        
    #     num_batch, num_ch, num_freq, num_data = data.shape
    #     f_iter_all = np.zeros((num_batch, num_freq, self.len_fun))
    #     f_iter_all = torch.tensor(f_iter_all, dtype=torch.float32).cuda()
        
    #     for jj in range(num_batch):
    #         if type(init_val) == type(None):
    #             f_iter = torch.tensor(np.zeros((self.len_fun, )), dtype=torch.float32).cuda()
    #         else:
    #             temp = init_val[jj, 0, :, :]
    #             f_iter = torch.tensor(self.matrix2vec_gpu(temp[np.newaxis, np.newaxis, :, :]), \
    #                                   dtype=torch.float32).cuda()[0,0,:]
    #         for ii in range(self.num_kappas):
    #             self.F = -torch.matmul(self.M, f_iter) + self.FG
    #             sol_forward = torch.matmul(self.Ainv[ii], self.F)
    #             temp1 = torch.matmul(self.S, sol_forward) - self.mean
    #             temp = torch.matmul(self.precision, temp1 - data[jj, 0, ii,:])
    #             F_adjoint = -torch.matmul(self.S.T, temp)
    #             sol_adjoint = torch.matmul(self.Ainv[ii], F_adjoint)
    #             grad_val = sol_adjoint
    #             gnorm = torch.max(torch.abs(grad_val))
    #             f_iter += -step_length*grad_val/gnorm
    #             f_iter_all[jj, ii, :] = f_iter
        
    #     return self.vec2matrix_gpu(f_iter_all)
    
    def inverting_cuda(self, data, init_val=None, step_length=torch.tensor(0.05, dtype=torch.float32)):        
        num_batch, num_ch, num_freq, num_data = data.shape
        # print("num_batch: ", num_batch)
        len_funR = np.int(self.len_fun/2)
        f_iter_all = np.zeros((num_batch, num_freq, len_funR))
        f_iter_all = torch.tensor(f_iter_all, dtype=torch.float32).cuda()
        
        if type(init_val) != type(None):
            init_val = F.interpolate(init_val, size=[self.sa, self.sa], mode='bicubic')
            init_val = torch.tensor(self.matrix2vec_gpu(init_val), dtype=torch.float32).cuda()
        for jj in range(num_batch):
            if type(init_val) == type(None):
                f_iter = torch.tensor(np.zeros((len_funR, )), dtype=torch.float32).cuda()
            else:
                f_iter= init_val[jj, 0, :]
            for ii in range(self.num_kappas):
                for mm in range(1):
                    f_iterRI = combineRITorch(f_iter)
                    self.F = torch.matmul(self.M, f_iterRI)
                    sol_forward = torch.matmul(self.Ainv[ii], self.F)
                    data_simulate = self.S@sol_forward
                    res_vec = data_simulate - data[jj, 0, ii,:]                    
                    F_adjoint = torch.matmul(self.S.T, res_vec)
                    sol_adjoint = torch.matmul(self.Ainv_adjoint[ii], F_adjoint)
                    grad_val = sol_adjoint[::2]
                    gnorm = torch.max(torch.abs(grad_val)) + 1e-15
                    f_iter += -step_length*grad_val/gnorm
                f_iter_all[jj, ii, :] = f_iter
        
        return self.vec2matrix_gpu(f_iter_all)


def gene_vec2matrix(V, device='cpu'):
    vertex_to_dof = np.int64(fe.vertex_to_dof_map(V))
    if device == 'cpu':
        def vec2matrix(x):
            # only available for P_1 element
            num, c, ll = x.shape
            xx = x[:, :, vertex_to_dof]
            num, c, chang = xx.shape
            chang = np.int(np.sqrt(chang))
            output = np.array(xx.reshape((num, c, chang, chang)))
            return output
    elif device == 'gpu':
        def vec2matrix(x):
            # only available for P_1 element
            num, c, ll = x.shape
            xx = x[:, :, vertex_to_dof]
            num, c, chang = xx.shape
            chang = np.int(np.sqrt(chang))
            output = xx.reshape((num, c, chang, chang))
            return output
        
    return vec2matrix


def gene_matrix2vec(V, device='cpu'):
    dof_to_vertex = np.int64(fe.dof_to_vertex_map(V))
    if device == 'cpu':
        def matrix2vec(x):
            # only available for P_1 element
            num, c, a, b = x.shape
            x = x.reshape((num, c, a*b))
            output = np.array(x[:, :, dof_to_vertex])
            # output = x[: ,:, dof_to_vertex]
            return output
    elif device == 'gpu':
        def matrix2vec(x):
            num, c, a, b = x.shape
            x = x.reshape((num, c, a*b))
            output = x[: ,:, dof_to_vertex]
            return output
    else:
        sys.exit("device must be cpu or gpu")
        
    return matrix2vec










