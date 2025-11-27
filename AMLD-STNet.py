#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import math


def Rho(x, dim):
    h = 1
    center_diff = (x[:, 2:] - x[:, :-2]) / (2 * h)
    prepend = (x[:, 1:2] - x[:, :1]) / h
    append = (x[:, -1:] - x[:, -2:-1]) / h
    center_diff = torch.cat((prepend, center_diff, append), dim=dim)
    return center_diff


class Gamma(nn.Module):
    def __init__(self, in_features, out_features, adj, bias=True):
        super(Gamma, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(support, self.adj)
        return output + self.bias if self.bias is not None else output


def build_tridiagonal_diag(n, a, b, c):
    diag_main = torch.diag(torch.full((n,), a))
    diag_super = torch.diag(torch.full((n - 1,), b), diagonal=1)
    diag_sub = torch.diag(torch.full((n - 1,), c), diagonal=-1)
    return diag_main + diag_super + diag_sub


class JointForceRestoring(nn.Module):
    def __init__(self, in_channels, output_channels, joints_dim, joints_dim_r, p=0, dropout=None):
        super(JointForceRestoring, self).__init__()
        self.attentions = GlobalGraph(in_channels, output_channels // 2, joints_dim, joints_dim_r, p)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.p = p
        self.joints_dim_r = joints_dim_r

    def forward(self, x, y):
        x = x.permute(0, 2, 3, 1)
        x_size = x.shape
        x = x.contiguous().view(-1, *x_size[2:])
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 3, 1)
        y_size = y.shape
        y = y.contiguous().view(-1, *y_size[2:])
        y = y.permute(0, 2, 1)
        x = self.attentions(x, y)
        x = x.permute(0, 2, 1).contiguous()
        if self.p == 0:
            x = x.view(*x_size)
        else:
            x = x.view(*x_size[:2], self.joints_dim_r, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.bn(x))
        return self.dropout(x) if self.dropout is not None else x


class GlobalGraph(nn.Module):
    def __init__(self, in_channels, inter_channels, joints_dim, joints_dim_r, p=0):
        super(GlobalGraph, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.g_channels = self.in_channels if self.inter_channels == self.in_channels // 2 else self.inter_channels
        assert self.inter_channels > 0
        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.g_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.concat_project = nn.Sequential(nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False))
        self.C_k = nn.Parameter(torch.zeros(joints_dim, joints_dim_r, dtype=torch.float))
        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)
        self.p = p

    def forward(self, x, y):
        batch_size = x.size(0)
        if self.p == 0:
            g_x = self.g(y).view(batch_size, self.g_channels, -1).permute(0, 2, 1)
        else:
            g_x = self.g(x).view(batch_size, self.g_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        phi_x = self.phi(y).view(batch_size, self.inter_channels, 1, -1)
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.expand(-1, -1, -1, w)
        phi_x = phi_x.expand(-1, -1, h, -1)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        attention = self.leakyrelu(f.view(b, h, w))
        if self.p == 0:
            attention = torch.add(self.softmax(attention), self.C_k)
        else:
            attention = torch.add(self.softmax(attention), self.C_k)
            attention = torch.einsum('ijk->ikj', attention)
        y = torch.matmul(attention, g_x).permute(0, 2, 1).contiguous()
        return y


class LDSTNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, time_dim, joints_dim, dropout, bias=True):
        super(LDSTNet, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        self.gcn_cd1 = LDNET(in_channels, joints_dim)
        self.Pool_1 = JointForcePooling(joints_dim, joints_dim - 6, in_channels, time_dim)
        self.gcn_cd2 = LDNET(in_channels, joints_dim - 6)
        self.att_cd2 = JointForceRestoring(in_channels, in_channels, joints_dim, joints_dim - 6, dropout=dropout)
        self.Pool_2 = JointForcePooling(joints_dim, joints_dim - 10, in_channels, time_dim)
        self.gcn_cd3 = LDNET(in_channels, joints_dim - 10)
        self.att_cd3 = JointForceRestoring(in_channels, in_channels, joints_dim, joints_dim - 10, dropout=dropout)
        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, (self.kernel_size[0], self.kernel_size[1]), (stride, stride),
                      padding), nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True))
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1)),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.residual = nn.Identity()
        self.prelu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        x_cd1, FA_1 = self.gcn_cd1(x)
        x_2 = self.Pool_1(x_cd1, FA_1)
        x_cd2, _ = self.gcn_cd2(x_2)
        x_cd2 = self.att_cd2(x_cd1, x_cd2)
        x_3 = self.Pool_2(x_cd1, FA_1)
        x_cd3, _ = self.gcn_cd3(x_3)
        x_cd3 = self.att_cd3(x_cd1, x_cd3)
        x = torch.cat((x, x_cd1, x_cd2, x_cd3), dim=1)
        x = self.tcn(x)
        x = x + res
        x = self.prelu(x)
        return x


class JointForceDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, bias=True):
        super(JointForceDecoder, self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                                   nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True))

    def forward(self, x):
        return self.block(x)


class JointForcePooling(nn.Module):
    def __init__(self, input_dim, output_dim, T, J):
        super(JointForcePooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = T
        self.J = J
        self.Nr = output_dim
        self.scalen_r = nn.Linear(T * J, output_dim, bias=False)
        self.prelu = nn.PReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Fn, FA):
        batch_size, C, T, N = Fn.shape
        F = self.prelu(torch.einsum('bijk,bjkl->bijl', Fn, FA))
        F = F.permute(0, 1, 3, 2)
        F = F.contiguous().view(batch_size, -1, N).permute(0, 2, 1)
        F = self.softmax(self.scalen_r(F))
        F_out = torch.einsum('bijk,bkl->bijl', Fn, F)
        return F_out


class AMLDSTNet(nn.Module):
    def __init__(self, input_channels, input_time_frame, output_time_frame, st_gcnn_dropout, joints_to_consider,
                 n_txcnn_layers, txc_kernel_size, txc_dropout, bias=True):
        super(AMLDSTNet, self).__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.joints_to_consider = joints_to_consider
        self.JointForceEncoder = nn.ModuleList()
        self.JointForceDecoder_layers = n_txcnn_layers
        self.JointForceDecoder = nn.ModuleList()
        self.JointForceEncoder.append(LDSTNet(5, 32, [1, 1], 1, input_time_frame, joints_to_consider, st_gcnn_dropout))
        self.JointForceEncoder.append(
            LDSTNet(32, input_channels, [1, 1], 1, input_time_frame, joints_to_consider, st_gcnn_dropout))
        self.JointForceDecoder.append(
            JointForceDecoder(input_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        for i in range(1, n_txcnn_layers):
            self.JointForceDecoder.append(
                JointForceDecoder(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        for gcn in self.JointForceEncoder:
            x = gcn(x)
        x = x.permute(0, 2, 1, 3)
        x = self.prelus[0](self.JointForceDecoder[0](x))
        for i in range(1, self.JointForceDecoder_layers):
            x = self.prelus[i](self.JointForceDecoder[i](x)) + x
        return x

