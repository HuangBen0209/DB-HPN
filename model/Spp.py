import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels=None):
        super(SpatialPyramidPooling, self).__init__()
        if levels is None:
            levels = [1, 2, 3, 4]
        self.levels = levels
    def forward(self, x):
        N, C, H, W = x.size()
        outputs = []
        for level in self.levels:
            kernel_size = (int(np.ceil(H / level)), int(np.ceil(W / level)))
            stride = kernel_size
            pooling = F.adaptive_max_pool2d(x, output_size=(level, level))
            outputs.append(pooling.view(N, -1))
        return torch.cat(outputs, dim=1)