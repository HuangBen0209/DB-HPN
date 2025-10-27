# -*- coding: utf-8 -*-

import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("./model/Temporal_shift/")
from model.Temporal_shift.cuda.shift import Shift
from model.stgcn_block import STGCNBlock
from model.stgcn_block import EnhancedSTGCNBlock
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)
def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = ((kernel_size - 1) // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)
    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal_(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        x_in = self.shift_in(x)
        x = self.temporal_linear(x_in)
        x = self.relu(x)
        x_out = self.shift_out(x)
        x = self.bn2(x_out)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = lambda x: x
        self.A = A
        num_nodess = A.shape[1]
        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'),
                                          requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels, requires_grad=True, device='cuda'),
                                        requires_grad=True)
        nn.init.constant_(self.Linear_bias, 0)
        self.Feature_Mask = nn.Parameter(torch.ones(1, num_nodess, in_channels, requires_grad=True, device='cuda'),
                                         requires_grad=True)
        nn.init.constant_(self.Feature_Mask, 0)
        self.bn = nn.BatchNorm1d(num_nodess * out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        index_array = np.zeros(num_nodess * in_channels)
        index_array = index_array.astype(np.int32)

        for i in range(num_nodess):  # 位移操作
            for j in range(in_channels):
                index_array[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (in_channels * num_nodess)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)
        index_array = np.zeros(num_nodess * out_channels)
        index_array = index_array.astype(np.int32)

        for i in range(num_nodess):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (
                        out_channels * num_nodess)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)
    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0, 2, 3, 1).contiguous()
        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)  # 位移操作
        x = x.view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous()  # nt,v,c
        x = x + self.Linear_bias
        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2)  # n,c,t,v
        x = x + self.res(x0)
        x = self.relu(x)
        return x
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)
    def forward(self, x):
        x_main = self.tcn1(self.gcn1(x))
        x_res = self.residual(x)
        x = x_main + x_res
        return self.relu(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Model(nn.Module):
    def __init__(self, num_class=15, num_nodes=25, num_person=1, graph='graph.ntu_rgb_d.Graph', graph_args=None,
                 in_channels=3, **kwargs):
        super(Model, self).__init__()

        if graph_args is None:
            graph_args = dict()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, device='cuda').detach().clone()
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9  # 时间卷积核大小为9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  # 9*25
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            STGCNBlock(64, 64, kernel_size, 1, name="Block1", **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, name="Block2", **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, name="Block3", **kwargs),
            STGCNBlock(64, 128, kernel_size, 2, name="Block4", **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, name="Block5", **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, name="Block6", **kwargs),
            STGCNBlock(128, 256, kernel_size, 2, name="Block7", **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, name="Block8", **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, name="Block9", **kwargs),
        ))
        base_channel = 64
        stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channel, kernel_size=(4, 1), stride=(4, 1)),
            LayerNorm(base_channel, eps=1e-6, data_format="channels_first")
        )
        self.enhanceSTGCN = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            EnhancedSTGCNBlock(64, 64, kernel_size, 1, name="Block1", **kwargs),
            EnhancedSTGCNBlock(64, 64, kernel_size, 1, name="Block2",**kwargs),
            EnhancedSTGCNBlock(64, 64, kernel_size, 1, name="Block3", **kwargs),
            EnhancedSTGCNBlock(64, 128, kernel_size, 2, name="Block4", **kwargs),
            EnhancedSTGCNBlock(128, 128, kernel_size, 1, name="Block5", **kwargs),
            EnhancedSTGCNBlock(128, 128, kernel_size, 1, name="Block6", **kwargs),
            EnhancedSTGCNBlock(128, 256, kernel_size, 2, name="Block7", **kwargs),
            EnhancedSTGCNBlock(256, 256, kernel_size, 1, name="Block8", **kwargs),
            EnhancedSTGCNBlock(256, 256, kernel_size, 1, name="Block9", **kwargs),
        ))
        self.shift_gcn_networks = nn.ModuleList((
            TCN_GCN_unit(in_channels, 64,A, residual=False),
            TCN_GCN_unit(64, 64, A),
            TCN_GCN_unit(64, 64, A),
            TCN_GCN_unit(64, 64, A),
            TCN_GCN_unit(64, 128, A, stride=2),
            TCN_GCN_unit(128, 128, A),
            TCN_GCN_unit(128, 128, A),
            TCN_GCN_unit(128, 256, A, stride=2),
            TCN_GCN_unit(256, 256, A),
            TCN_GCN_unit(256, 256, A)
        ))

        self.fc_st = nn.Linear(256, num_class)
        self.fc_shift = nn.Linear(256, num_class)
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.fc_fusion = nn.Linear(num_class, num_class)
        nn.init.normal_(self.fc_fusion.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        x_stgcn = x.clone()
        x_shiftgcn = x.clone()
        st_gcn_A = self.A.clone()

        for stgcn in self.enhanceSTGCN:
            x_stgcn = stgcn(x_stgcn, st_gcn_A)
        for shiftgcn in self.shift_gcn_networks:
            x_shiftgcn = shiftgcn(x_shiftgcn)

        c_new = x_stgcn.size(1)
        x_stgcn = x_stgcn.view(N, M, c_new, -1)
        x_stgcn = x_stgcn.mean(3).mean(1)
        x_stgcn = self.fc_st(x_stgcn)

        c_new = x_shiftgcn.size(1)
        x_shiftgcn = x_shiftgcn.view(N, M, c_new, -1)
        # 先在维度3上取最大值，再在维度1上取最大值
        x_shiftgcn = x_shiftgcn.mean(3).mean(1)

        x_shiftgcn = self.fc_shift(x_shiftgcn)
        prob_st = F.softmax(x_stgcn, dim=1)  # [N, num_classes]
        prob_shift = F.softmax(x_shiftgcn, dim=1)
        fusion_weights = F.softmax(self.fusion_weights, dim=0)  #
        fused_prob = fusion_weights[0] * prob_st + fusion_weights[1] * prob_shift
        fused_log_prob = torch.log(fused_prob + 1e-8)  # 避免 log(0) 报错
        return prob_st,prob_shift,fused_log_prob ,fusion_weights # 用于 NLLLoss
