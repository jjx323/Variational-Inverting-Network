#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .DnCNN import DnCNN
from .UNet import UNet

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net


class VDNU(nn.Module):
    def __init__(self, in_channels, activation='relu', act_init=0.01, wf=64, dep_S=5, dep_U=4,
                                                                                   batch_norm=True):
        super(VDNU, self).__init__()
        net1 = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, batch_norm=batch_norm,
                                                           activation=activation, act_init=act_init)
        self.DNet = weight_init_kaiming(net1)
        net2 = DnCNN(in_channels, in_channels*2, dep=dep_S, num_filters=64, activation=activation,
                                                                                  act_init=act_init)
        self.SNet = weight_init_kaiming(net2)

    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma


class VDNU_Invert(nn.Module):
    def __init__(self, in_chn_U, out_chn_U, in_chn_S, out_chn_S, activation='relu', \
                 act_init=0.01, wf=64, dep_S=5, dep_U=4, batch_norm=True, \
                 u_dim=128):
        super(VDNU_Invert, self).__init__()
        net1 = UNet(in_chn_U, out_chn_U, wf=wf, depth=dep_U, batch_norm=batch_norm,
                                           activation=activation, act_init=act_init)
        self.DNet = weight_init_kaiming(net1)
        net2 = DnCNN(in_chn_S, out_chn_S, dep=dep_S, num_filters=64, activation=activation,
                                                                     act_init=act_init)
        self.SNet = weight_init_kaiming(net2)
        self.in_chn_S, self.out_chn_S = in_chn_S, out_chn_S
        self.in_chn_U, self.out_chn_U = out_chn_U, out_chn_U
        
        self.u_dim = u_dim

    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            x = F.interpolate(x, size=[self.u_dim, self.u_dim], mode='bicubic')
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
            return phi_sigma


class VDNU_Invert_Null(nn.Module):
    def __init__(self, in_chn_U, out_chn_U, in_chn_S, out_chn_S, \
                 activation='relu', in_chn_I=None, out_chn_I=None, \
                 act_init=0.01, wf=64, dep_S=5, dep_U=4, dep_I=4, batch_norm=True, \
                 forward_op=None, data_dim=20, u_dim=128, repeat_number=1,\
                 freq_used=[0,10,20,23]):
        super(VDNU_Invert_Null, self).__init__()
        net1 = UNet(in_chn_U, out_chn_U, wf=wf, depth=dep_U, batch_norm=batch_norm,
                                           activation=activation, act_init=act_init)
        self.forward_op = forward_op
        if in_chn_I == None:
            in_chn_I = in_chn_U*3
            out_chn_I = out_chn_U
        net2 = UNet(in_chn_I, out_chn_I, wf=wf, depth=dep_I, batch_norm=batch_norm, \
                    activation=activation, act_init=act_init)
        net3 = UNet(in_chn_I, out_chn_I, wf=wf, depth=dep_I, batch_norm=batch_norm, \
                    activation=activation, act_init=act_init)
        self.DNet = weight_init_kaiming(net1)
        self.INet = weight_init_kaiming(net2)
        self.INet2 = weight_init_kaiming(net3)
        net4 = DnCNN(in_chn_S, out_chn_S, dep=dep_S, num_filters=64, activation=activation,
                                                                     act_init=act_init)
        self.SNet = weight_init_kaiming(net4)
        self.in_chn_S, self.out_chn_S = in_chn_S, out_chn_S
        self.in_chn_U, self.out_chn_U = out_chn_U, out_chn_U
        self.in_chn_I, self.out_chn_I = out_chn_I, out_chn_I
        
        self.data_dim = data_dim
        self.u_dim = u_dim
        self.repeat_number = repeat_number
        self.freq_used = freq_used
        
    def for_2U(self, phi_Z):
        mu = phi_Z[:, :np.int(self.out_chn_U/2), :, :]
        if self.forward_op.cuda == True:
            est_u = self.forward_op.inverting_cuda(mu)[:, self.freq_used, :, :]
        else:
            est_u = self.forward_op.inverting(mu)[:, self.freq_used, :, :]
        est_u = F.interpolate(est_u, size=[self.u_dim, self.u_dim], mode='bicubic')
        # cov = phi_Z[:, np.int(self.out_chn_U/2):, :, :]
        cov = F.interpolate(phi_Z, size=[self.u_dim, self.u_dim], mode='bicubic')
        out = torch.cat((est_u, cov), 1)
        return out
    
    def for_2Uinit(self, phi_Z, init_val):
        mu = phi_Z[:, :np.int(self.out_chn_U/2), :, :]
        if self.forward_op.cuda == True:
            est_u = self.forward_op.inverting_cuda(mu, init_val)[:, self.freq_used, :, :]
        else:
            est_u = self.forward_op.inverting(mu, init_val)[:, self.freq_used, :, :]
        est_u = F.interpolate(est_u, size=[self.u_dim, self.u_dim], mode='bicubic')
        # cov = phi_Z[:, np.int(self.out_chn_U/2):, :, :]
        cov = F.interpolate(phi_Z, size=[self.u_dim, self.u_dim], mode='bicubic')
        out = torch.cat((est_u, cov), 1)
        return out
    
    def restrict(self, x):
        # print('repeat_number = ', self.repeat_number)
        # print('x.shape = ', x.shape)
        return x[:,:,0:-1:self.repeat_number,:]
    
    def splitRI(self, x):
        ## split the real and imaginary part of the data
        b, c, l, r = x.shape
        xx = torch.empty_like(x)
        r_half = np.int(r/2)
        xx[:, :, :, :r_half], xx[:, :, :, r_half:] = x[:, :, :, ::2], x[:, :, :, 1::2]
        return xx
    
    def combineRI(self, x):
        b, c, l, r = x.shape
        xx = torch.empty_like(x)
        rr = np.int(r/2)
        for i in range(l):
            xx[:, :, i, ::2], xx[:, :, i, 1::2] = x[:, :, i, :rr], x[:, :, i, rr:]
        return xx
    
    def DNet2(self, x):
        xx = self.splitRI(x)
        phi_Z = self.DNet(xx)
        phi_Z = self.combineRI(phi_Z) 
        return phi_Z

    def forward(self, x, mode='train_DNet_SNet'):
        if mode.lower() == 'train_dnet_snet':
            phi_Z = self.DNet2(x)
            phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
            # phi_Z = self.restrict(phi_Z)
            # phi_sigma = self.restrict(phi_sigma)
            return phi_Z, phi_sigma
        elif mode.lower() == 'train_inet':
            with torch.set_grad_enabled(False):
                phi_Z = self.DNet2(x)
                data = self.restrict(phi_Z)
                # dataT = F.interpolate(data, size=[self.u_dim, self.u_dim], mode='bicubic')
                # print(dataT.shape)
            with torch.set_grad_enabled(True):
                # phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
                # phi_sigma = self.restrict(phi_sigma)
                # phi_Z = self.INet(est_u)
                phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
                phi_sigma = self.restrict(phi_sigma)
                est_u= self.for_2U(data)
                # out = torch.cat((est_u, dataT[:,0,:,:]), 1)
                # print(est_u.shape)
                phi_Z = self.INet(est_u)
                # phi_Z = self.INet(est_u)
                # init_val = phi_Z[:, :self.in_chn_S, :, :]
                # est_u = self.for_2Uinit(data, init_val)
                # phi_Z = self.INet2(est_u)
                # phi_Z = self.INet(est_u)
            return phi_Z, phi_sigma, est_u
        elif mode.lower() == 'train_all':
            # phi_Z = self.DNet(x)
            # phi_Z = self.restrict(phi_Z)
            # phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
            # phi_sigma = self.restrict(phi_sigma)
            # est_u= self.for_2U(phi_Z)
            # phi_Z = self.INet(est_u)
            phi_Z = self.DNet2(x)
            data = self.restrict(phi_Z)
            # dataT = F.interpolate(data, size=[self.u_dim, self.u_dim], mode='bicubic')
            phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
            phi_sigma = self.restrict(phi_sigma)
            est_u= self.for_2U(data)
            # out = torch.cat((est_u, dataT), 1)
            # phi_Z = self.INet(out)
            phi_Z = self.INet(est_u)
            # init_val = phi_Z[:, :self.in_chn_S, :, :]
            # est_u = self.for_2Uinit(data, init_val)
            # phi_Z = self.INet2(est_u)
            # phi_Z = self.INet(est_u)
            return phi_Z, phi_sigma, est_u
        elif mode.lower() == 'test':
            # phi_Z = self.DNet(x)
            # phi_Z = self.restrict(phi_Z)
            # phi_Z = self.for_2U(phi_Z)
            # phi_Z = self.INet(phi_Z)
            phi_Z = self.DNet2(x)
            data = self.restrict(phi_Z)
            # dataT = F.interpolate(data, size=[self.u_dim, self.u_dim], mode='bicubic')
            est_u= self.for_2U(data)
            # out = torch.cat((est_u, dataT), 1)
            # phi_Z = self.INet(out)
            phi_Z = self.INet(est_u)
            # init_val = phi_Z[:, :self.in_chn_S, :, :]
            # est_u = self.for_2Uinit(data, init_val)
            # phi_Z = self.INet2(est_u)
            # phi_Z = self.INet(est_u)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
            phi_sigma = self.restrict(phi_sigma)
            return phi_sigma
        # elif mode.lower() == 'doubletest':
        #     phi_Z = self.DNet(x)
        #     data = self.restrict(phi_Z)
        #     phi_sigma = self.SNet(x[:, :self.in_chn_S, :, :])
        #     phi_sigma = self.restrict(phi_sigma)
        #     est_u= self.for_2U(data)
        #     phi_Z = self.INet(est_u)
        #     init_val = phi_Z[:, :self.in_chn_S, :, :]
        #     est_u = self.for_2Uinit(data, init_val)
        #     phi_Z = self.INet(est_u)
        #     return phi_Z, phi_sigma, est_u



