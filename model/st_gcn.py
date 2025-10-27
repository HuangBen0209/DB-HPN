import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph import Graph
from model.stgcn_block import STGCNBlock
from model.stgcn_block import EnhancedSTGCNBlock




class STGCN(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, num_nodes,
                 edge_importance_weighting, use_residual=True, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args, num_nodes=num_nodes)
        if graph_args['strategy'] == 'adaptive':  #让邻接矩阵参与更新
            self.A = self.graph.A
        else:
            A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False) #固定邻接矩阵，不参与更新
            self.register_buffer('A', A)

        # 构建网络
        if graph_args['strategy'] == 'adaptive':  # 空间卷积核大小为邻接矩阵的个数
            spatial_kernel_size = self.A.size(0)
        else:
            spatial_kernel_size = self.A.size(0)

        temporal_kernel_size = 9  # 时间卷积核大小为9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  #
        # 更新 self.data_bn 的输入维度
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes) #对每一帧的所有关节、所有通道进行归一化
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            STGCNBlock(64, 64, kernel_size, 1, residual=use_residual,name="Block1", **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, residual=use_residual, name="Block2",**kwargs),
            STGCNBlock(64, 64, kernel_size, 1, residual=use_residual, name="Block3",**kwargs),
            STGCNBlock(64, 128, kernel_size, 2, residual=use_residual, name="Block4",**kwargs),
            STGCNBlock(128, 128, kernel_size, 1, residual=use_residual,name="Block5", **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, residual=use_residual,name="Block6", **kwargs),
            STGCNBlock(128, 256, kernel_size, 2, residual=use_residual,name="Block7", **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, residual=use_residual,name="Block8", **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, residual=use_residual,name="Block9", **kwargs)
        ))
        self.enhanceSTGCN = nn.ModuleList((
            EnhancedSTGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            EnhancedSTGCNBlock(64, 64, kernel_size, 1, name="Block1", **kwargs),
            EnhancedSTGCNBlock(64, 64, kernel_size, 1, name="Block2", **kwargs),
            EnhancedSTGCNBlock(64, 64, kernel_size, 1, name="Block3", **kwargs),
            EnhancedSTGCNBlock(64, 128, kernel_size, 2, name="Block4", **kwargs),
            EnhancedSTGCNBlock(128, 128, kernel_size, 1, name="Block5", **kwargs),
            EnhancedSTGCNBlock(128, 128, kernel_size, 1, name="Block6", **kwargs),
            EnhancedSTGCNBlock(128, 256, kernel_size, 2, name="Block7", **kwargs),
            EnhancedSTGCNBlock(256, 256, kernel_size, 1, name="Block8", **kwargs),
            EnhancedSTGCNBlock(256, 256, kernel_size, 1, name="Block9", **kwargs),
        ))
        # 初始化边缘重要性加权的参数
        if edge_importance_weighting:  #边重要性加权，为每一层stgcnblock的邻接矩阵设置一个边重要性参数
            if graph_args['strategy'] == 'adaptive':
                self.edge_importance = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A.size()))
                    for i in self.st_gcn_networks
                ])
            else:
                self.edge_importance = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A.size()))
                    for i in self.st_gcn_networks
                ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 全连接层
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)


    def forward(self, x):
        # 数据预处理，将 NCTVM 形状转换为 NCTVM
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # 前向传播
        for gcn, importance in zip(self.enhanceSTGCN, self.edge_importance):
            if isinstance(self.A, nn.Parameter):
                A = self.A
            else:
                A = self.A * importance
            x = gcn(x, A)


        # 全局池化
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        # 预测
        x = self.fcn(x.unsqueeze(-1).unsqueeze(-1))
        x = x.squeeze(-1).squeeze(-1)

        return x
