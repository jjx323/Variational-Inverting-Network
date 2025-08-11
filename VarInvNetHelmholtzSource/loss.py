#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
import torch.nn.functional as F
from math import pi, log
from VarInvNetHelmholtzSource.utils import LogGamma
import numpy as np
import scipy.sparse.linalg as spsl
from scipy.sparse import coo_matrix

log_gamma = LogGamma.apply

# clip bound
log_max = log(1e4)
log_min = log(1e-8)

## UNet learns the mapping from noisy image to residual = im_noisy - im_gt
def loss_fn_residual(out_denoise, out_sigma, im_noisy, im_gt, sigmaMap, eps2, radius=3):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
        mask: (N,)  array
    '''
    C = im_gt.shape[1]
    p = 2*radius+1
    p2 = p**2
    alpha0 = 0.5 * torch.tensor([p2-2]).type(sigmaMap.dtype).to(device=sigmaMap.device)
    beta0 = 0.5 * p2 * sigmaMap

    # parameters predicted of Gaussain distribution
    out_denoise[:, C:,].clamp_(min=log_min, max=log_max)
    # if err_clip:
        # out_denoise[:, :C,].clamp_(min=err_min, max=err_max)
    err_mean = out_denoise[:, :C,]
    m2 = torch.exp(out_denoise[:, C:,])   # variance

    # parameters predicted of Inverse Gamma distribution
    out_sigma.clamp_(min=log_min, max=log_max)
    log_alpha = out_sigma[:, :C,]
    alpha = torch.exp(log_alpha)
    log_beta = out_sigma[:, C:,]
    alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL divergence for Gauss distribution
    m2_div_eps = torch.div(m2, eps2)
    err_mean_gt = im_noisy - im_gt
    kl_gauss =  0.5*(err_mean-err_mean_gt)**2/eps2 + 0.5*(m2_div_eps - 1 - torch.log(m2_div_eps))
    loss_kl_gauss = torch.mean(kl_gauss)

    # KL divergence for Inv-Gamma distribution
    kl_Igamma = (alpha-alpha0)*torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha)) + \
                               alpha0*(log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha
    loss_kl_Igamma = torch.mean(kl_Igamma)

    # likelihood of im_gt
    lh = 0.5 * log(2*pi) + 0.5 * (log_beta - torch.digamma(alpha)) + \
                                                             0.5 * (err_mean**2+m2) * alpha_div_beta
    loss_lh = torch.mean(lh)

    loss = loss_lh + loss_kl_gauss + loss_kl_Igamma

    return loss, loss_lh, loss_kl_gauss, loss_kl_Igamma

## UNet learns the mapping from noisy image to clean image
def loss_fn_clean(out_denoise, out_sigma, im_noisy, im_gt, sigmaMap, eps2, radius=3, \
                  forward_op=None, kl_gauss_type='full'):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
        mask: (N,)  array
    '''
    C = im_gt.shape[1]
    p = 2*radius+1
    p2 = p**2
    alpha0 = 0.5 * torch.tensor([p2-2]).type(sigmaMap.dtype).to(device=sigmaMap.device)
    beta0 = 0.5 * p2 * sigmaMap

    # parameters predicted of Gaussain distribution
    out_denoise[:, C:,].clamp_(min=log_min, max=log_max)
    # if err_clip:
        # out_denoise[:, :C,].clamp_(min=err_min, max=err_max)
    clean_mean = out_denoise[:, :C,]
    m2 = torch.exp(out_denoise[:, C:,])   # variance
          
    # parameters predicted of Inverse Gamma distribution
    out_sigma.clamp_(min=log_min, max=log_max)
    log_alpha = out_sigma[:, :C,]
    alpha = torch.exp(log_alpha)
    log_beta = out_sigma[:, C:,]
    alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL divergence for Gauss distribution
    if kl_gauss_type == 'MC':
        ep_var = torch.randn_like(clean_mean)
        z_tilde = clean_mean + torch.sqrt(m2)*ep_var
        m2_div_eps = torch.div(m2, eps2)
        a1 = 0.5*((torch.div(z_tilde - clean_mean, torch.sqrt(m2)))**2)
        a2 = 0.5*((torch.div(z_tilde - im_gt, eps2))**2)
        kl_gauss = -0.5*torch.log(m2_div_eps) + a2 - a1
        loss_kl_gauss = torch.mean(kl_gauss)
    else:
        m2_div_eps = torch.div(m2, eps2)
        kl_gauss =  0.5*(clean_mean - im_gt)**2/eps2 + 0.5*(m2_div_eps - 1 - torch.log(m2_div_eps))
        loss_kl_gauss = torch.mean(kl_gauss)

    # KL divergence for Inv-Gamma distribution
    kl_Igamma = (alpha-alpha0)*torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha)) + \
                               alpha0*(log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha
    loss_kl_Igamma = torch.mean(kl_Igamma)

    # likelihood of im_gt
    if type(forward_op) == type(None):
         # likelihood of im_gt calculated by explicit formula (forward_op == Id)
        err_mean = im_noisy - clean_mean
        lh = 0.5 * log(2*pi) + 0.5 * (log_beta - torch.digamma(alpha)) + \
                               0.5 * (err_mean**2+m2) * alpha_div_beta
        loss_lh = torch.mean(lh)
    else:
        # likelihood of im_gt calculated by MC estimation
        ep_var = torch.randn_like(clean_mean)
        z_tilde = clean_mean + torch.sqrt(m2)*ep_var
        lh = 0.5 * log(2*pi) + 0.5 * (log_beta - torch.digamma(alpha)) + \
                              0.5 * alpha_div_beta * (im_noisy - forward_op(z_tilde))**2
        #print(z_tilde.shape)
        loss_lh = torch.mean(lh)
    

    loss = loss_lh + loss_kl_gauss + loss_kl_Igamma

    return loss, loss_lh, loss_kl_gauss, loss_kl_Igamma


def loss_fn_clean_FEM(out_denoise, out_sigma, im_noisy, im_gt, sigmaMap, eps2, \
                      geneMf, matrix2vec, vec2matrix, radius=3, dx=0.1, \
                      forward_op=None, kl_gauss_type='full'):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
        mask: (N,)  array
    '''
    C = im_gt.shape[1]
    p = 2*radius+1
    p2 = p**2
    alpha0 = 0.5 * torch.tensor([p2-2]).type(sigmaMap.dtype).to(device=sigmaMap.device)
    beta0 = 0.5 * p2 * sigmaMap

    # parameters predicted of Gaussain distribution
    out_denoise[:, C:,].clamp_(min=log_min, max=log_max)
    # if err_clip:
        # out_denoise[:, :C,].clamp_(min=err_min, max=err_max)
    clean_mean = out_denoise[:, :C,]
    m2 = torch.exp(out_denoise[:, C:,])   # variance
            
    # parameters predicted of Inverse Gamma distribution
    out_sigma.clamp_(min=log_min, max=log_max)
    log_alpha = out_sigma[:, :C,]
    alpha = torch.exp(log_alpha)
    log_beta = out_sigma[:, C:,]
    alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL divergence for Gauss distribution
    if kl_gauss_type == 'MC':
        # this implementation is just suitable for regular mesh (like finite difference)
        ep_var = torch.randn_like(clean_mean)
        z_tilde = clean_mean + torch.sqrt(m2)*ep_var*dx
        # ep_var = torch.randn_like(clean_mean)
        # Mf_list, M, M_diag, Minvhalf, V = geneMf(ep_var.detach().cpu().numpy())
        # m1 = matrix2vec(torch.sqrt(m2))
        # M_diag = torch.tensor(M_diag, dtype=torch.float32).cuda()
        # for ii in range(len(Mf_list)):
        #     coo = coo_matrix(Mf_list[ii])
        #     values = coo.data
        #     indices = np.vstack((coo.row, coo.col))
        #     iii = torch.LongTensor(indices)
        #     vvv = torch.FloatTensor(values)
        #     shape = coo.shape
        #     temp = torch.sparse.FloatTensor(iii, vvv, torch.Size(shape)).cuda()
        #     m1[ii, 0, :] = torch.div(torch.matmul(temp, m1[ii, 0, :]), M_diag)
        # z_tilde = clean_mean + vec2matrix(m1)
        
        m2_div_eps = torch.div(m2, eps2)
        a1 = 0.5*((torch.div(z_tilde - clean_mean, torch.sqrt(m2)))**2)*dx*dx
        a2 = 0.5*((torch.div(z_tilde - im_gt, eps2))**2)*dx*dx
        kl_gauss = -0.5*torch.log(m2_div_eps) + a2 - a1
        loss_kl_gauss = torch.mean(kl_gauss)
    else:
        m2_div_eps = torch.div(m2, eps2)
        kl_gauss =  0.5*(clean_mean - im_gt)**2/eps2 + 0.5*(m2_div_eps - 1 - torch.log(m2_div_eps))
        loss_kl_gauss = torch.mean(kl_gauss)

    # KL divergence for Inv-Gamma distribution
    # print(alpha0.shape, log_beta.shape, beta0.shape, alpha_div_beta.shape, alpha.shape)
    kl_Igamma = (alpha-alpha0)*torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha)) + \
                               alpha0*(log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha
    loss_kl_Igamma = torch.mean(kl_Igamma)

    # likelihood of im_gt
    if type(forward_op) == type(None):
         # likelihood of im_gt calculated by explicit formula (forward_op == Id)
        err_mean = (im_noisy - clean_mean)
        lh = 0.5 * log(2*pi) + 0.5 * (log_beta - torch.digamma(alpha)) + \
                               0.5 * (err_mean**2+m2) * alpha_div_beta
        loss_lh = torch.mean(lh)
    else:
        # ep_var = torch.randn_like(clean_mean)
        # Mf_list, M, M_diag, Minvhalf, V = geneMf(ep_var.detach().cpu().numpy())
        # m1 = matrix2vec(torch.sqrt(m2))
        # M_diag = torch.tensor(M_diag, dtype=torch.float32).cuda()
        # for ii in range(len(Mf_list)):
        #     coo = coo_matrix(Mf_list[ii])
        #     values = coo.data
        #     indices = np.vstack((coo.row, coo.col))
        #     iii = torch.LongTensor(indices)
        #     vvv = torch.FloatTensor(values)
        #     shape = coo.shape
        #     temp = torch.sparse.FloatTensor(iii, vvv, torch.Size(shape)).cuda()
        #     m1[ii, 0, :] = torch.div(torch.matmul(temp, m1[ii, 0, :]), M_diag)
        # z_tilde = clean_mean + vec2matrix(m1)
        # likelihood of im_gt calculated by MC estimation
        ep_var = torch.randn_like(clean_mean)
        z_tilde = clean_mean + torch.sqrt(m2)*ep_var*dx
        
        _, _, num_freq, _ = im_noisy.shape
        num_random_choice = 2
        index_random = sorted(np.random.choice(num_freq, num_random_choice, replace=False))
        index_random = torch.tensor(index_random, dtype=torch.long)
        rate_random = torch.tensor(num_freq/num_random_choice, dtype=torch.float32).cuda()
               
        lh_1 = 0.5 * log(2*pi) + 0.5 * (log_beta - torch.digamma(alpha)) 
        lh_2 = 0.5*alpha_div_beta[:,:,index_random,:]*rate_random*(\
                    im_noisy[:,:,index_random,:] - forward_op(z_tilde, index_random))**2
        loss_lh = torch.mean(lh_1) + torch.mean(lh_2)
        
        # lh = 0.5 * log(2*pi) + 0.5 * (log_beta - torch.digamma(alpha)) + \
             # 0.5 * alpha_div_beta * (im_noisy - forward_op(z_tilde))**2
        
        # loss_lh = torch.mean(lh)
    

    loss = loss_lh + loss_kl_gauss + loss_kl_Igamma

    return loss, loss_lh, loss_kl_gauss, loss_kl_Igamma




















