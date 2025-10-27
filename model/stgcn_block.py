import torch
import torch.nn as nn
from model.conv_temporal_graphical import ConvTemporalGraphical

def no_residual(x):
    return 0

def identity_residual(x):
    return x

class STGCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 name="STGCNBlock"):
        super().__init__()
        self.name = name
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1), # 时间卷积核大小为 kernel_size[0]，空间卷积核大小为 1
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = no_residual
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = identity_residual
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    # # 先设置torch打印选项，方便整体数字精度控制
    # torch.set_printoptions(precision=6, sci_mode=False)
    def forward(self, x, A):
        # # 把 A 的第0个矩阵转成 numpy
        # A_np = A[0].cpu().detach().numpy()
        # # 设置 numpy 的打印格式，保证每个数字占固定宽度，方便观察对齐
        # np.set_printoptions(precision=6, suppress=True, linewidth=120, formatter={'float': '{:8.6f}'.format})

        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        # with open('adjacency_log.txt', 'a') as f:
        #     f.write(f'[{self.name}] 形状: {A.shape}\n')
        #     f.write(f'[{self.name}] 值 (第一分支):\n{A_np}\n')
        return self.relu(x)

class TemporalInception(nn.Module):
    """时序多尺度模块 (时间多尺度)"""
    def __init__(self, in_channels, out_channels, name="TemporalInception"):
        super().__init__()
        self.name = name
        # 多分支时序卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 1)),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1),
                      dilation=1, padding=(1, 0))
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 1)),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(5, 1),
                      dilation=2, padding=(4, 0))
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 1)),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(7, 1),
                      dilation=3, padding=(9, 0))
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        )
        # 输出融合
        self.post_fusion = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, C, T, V)
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch_pool(x)
        ], dim=1)
        return self.post_fusion(out)
class EnhancedSTGCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 use_multi_scale=True,  # 是否使用多尺度
                 name="EnhancedSTGCNBlock"):
        super().__init__()
        self.name = name
        self.use_multi_scale = use_multi_scale
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(
            in_channels, out_channels, kernel_size[1]
        )
        # ===== 时间建模 =====
        self.tcn = nn.Sequential(
            # 时序多尺度模块替换原始TCN
            TemporalInception(out_channels, out_channels, name=f"{name}_TemporalInception"),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),  # 时间卷积核大小为 kernel_size[0]，空间卷积核大小为 1
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        # ===== 残差连接 =====
        if not residual:
            self.residual = no_residual
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = identity_residual
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else nn.Identity()

    def forward(self, x, A):
        res = self.residual(x)

        # 处理邻接矩阵输入
        if self.use_multi_scale:
            # 为多尺度准备邻接矩阵列表
            # 假设A是包含多个尺度的邻接矩阵的列表
            A_multi = A
        else:
            # 单尺度时使用原始邻接矩阵
            A_multi = A

        # 空间建模
        x = self.gcn(x, A_multi)
        x = self.dropout(x)

        # 时间建模
        x = self.tcn(x) + res

        return self.relu(x)