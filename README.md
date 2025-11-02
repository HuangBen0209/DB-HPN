# ST-Shift-GCN Fusion: Multi-Stream Fusion Model for Skeleton-Based Action Recognition

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.7%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>


## Project Overview

ST-Shift-GCN Fusion is a skeleton-based action recognition framework that uses Spatio-Temporal Shift Graph Convolutional Networks (ST-Shift-GCN) for feature extraction and employs **probability fusion at the softmax layer** to improve recognition accuracy.

### Core Features

- **Multi-Dataset Support**: Works with mainstream skeleton action datasets including MSRAction3D, UTKinectAction3D, and more
- **Spatio-Temporal Fusion**: Combines spatial graph convolution with temporal shift operations to capture spatio-temporal features effectively
- **Probability Fusion Strategy**: Enhances model performance by fusing multi-stream probabilities at the softmax layer
- **Reproducibility**: Complete random seed setup ensures fully reproducible experimental results
- **Flexible Configuration**: Modular design supports different graph structure configurations and training parameters

## Environment Requirements

### Core Dependencies

```
Python 3.7+
PyTorch 1.8+
NumPy
```

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/HuangBen0209/DB-HPN.git
cd DB-HPN
```

2. Install dependencies:

```bash
pip install torch numpy
```

## Quick Start

### Data Preparation

1. Organize your dataset in this structure:

```
dataset/
├── MSRAction3D_origin/
│   └── joint/
│       ├── train_data.npy
│       ├── train_label.npy
│       ├── test_data.npy
│       └── test_label.npy
└── UTKinectAction3D_origin/
    └── joint/
        ├── train_data.npy
        └── ...
```

### Model Training

Run the main training script:

```bash
python main_fusion.py
```

### Key Configuration Parameters

Adjust these parameters in `main_fusion.py`:

```python
# Training parameters
lr = 0.001
num_epochs = 300
batch_size_list = [16]
run_seeds = [1]  # Random seeds

# Model parameters
use_residual = True
edge_importance_weighting = True
patience = 30  # Early stopping patience
```

## Project Structure

```
DB-HPN/
├── main_fusion.py                 # Main training and testing script
├── model/
│   └── st_shift_gcn_fusion.py     # ST-Shift-GCN fusion model definition
├── train/
│   └── train_model_fusion.py      # Trainer implementation
├── test/
│   └── test_model_fusion.py       # Tester implementation
├── graph/                         # Graph structure definitions
│   ├── msr.py                    # MSR dataset graph structure
│   └── ut.py                     # UT dataset graph structure
├── dataset/                       # Dataset directory
└── README.md
```

## Supported Datasets

| Dataset             | Joints | Action Classes | Graph Layout |
| ------------------- | ------ | -------------- | ------------ |
| MSRAction3D         | 20     | 20             | msr          |
| UTKinectAction3D    | 20     | 10             | ut           |
| Florence3DActions   | 15     | 9              | florence     |
| HanYueDailyAction3D | 25     | 15             | ntu-rgb+d    |

## Model Architecture

### ST-Shift-GCN Features

- **Spatial Modeling**: Uses graph convolutional networks to capture spatial relationships between joints
- **Temporal Modeling**: Processes time-series data through shift operations
- **Multi-Stream Fusion**: Fuses probability outputs from multiple data streams at the softmax layer
- **Adaptive Graph Structures**: Supports specialized graph layout strategies for different datasets

### Fusion Strategy

This project uses **softmax-level probability fusion**, performing weighted fusion of prediction probabilities from different streams to significantly improve final recognition accuracy.

## Experimental Results

Training automatically generates log files in `log_YYYY-MM-DD_HH_MM.txt` format, containing:

- Model configuration and hyperparameters
- Training loss and accuracy per epoch
- Final accuracy on test set
- Detailed multi-stream fusion results

## Contributing

We welcome community contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

### Contribution Guidelines

- Follow existing code style
- Add appropriate comments and documentation
- Update README.md for significant changes

## Contact

For questions or suggestions:

- Create an [Issue](https://github.com/HuangBen0209/DB-HPN/issues)
- Email the project maintainer

## Acknowledgments

Thanks to all developers who contributed to this project.


# ST-Shift-GCN Fusion: 基于骨骼动作识别的多流融合模型

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.7%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>


## 📋 项目概述

ST-Shift-GCN Fusion 是一个基于骨骼数据的动作识别框架，采用时空移位图卷积网络（ST-Shift-GCN）进行特征提取，并在 **softmax层进行概率融合** 以提高识别准确率。

### ✨ 核心特性

- **多数据集支持**: 支持 MSRAction3D、UTKinectAction3D 等多个主流骨骼动作数据集
- **时空融合**: 结合空间图卷积和时间移位操作，有效捕捉时空特征
- **概率融合策略**: 在 softmax 层进行多流概率融合，提升模型性能
- **可复现性**: 完整的随机种子设置，确保实验结果完全可复现
- **灵活配置**: 模块化设计，支持不同的图结构配置和训练参数

## 🛠 环境要求

### 核心依赖

```
Python 3.7+
PyTorch 1.8+
NumPy
```

### 安装步骤

1. 克隆项目：

```bash
git clone https://github.com/HuangBen0209/DB-HPN.git
cd DB-HPN
```

2. 安装依赖：

```bash
pip install torch numpy
```

## 🚀 快速开始

### 数据准备

1. 确保数据集按照以下结构组织：

```
dataset/
├── MSRAction3D_origin/
│   └── joint/
│       ├── train_data.npy
│       ├── train_label.npy
│       ├── test_data.npy
│       └── test_label.npy
└── UTKinectAction3D_origin/
    └── joint/
        ├── train_data.npy
        └── ...
```

### 模型训练

运行主训练脚本：

```bash
python main_fusion(2).py
```

### 关键配置参数

在 `main_fusion(2).py` 中调整以下参数：

```python
# 训练参数
lr = 0.001
num_epochs = 300
batch_size_list = [16]
run_seeds = [1]  # 随机种子

# 模型参数
use_residual = True
edge_importance_weighting = True
patience = 30  # 早停法耐心值
```

## 📁 项目结构

```
DB-HPN/
├── main_fusion(2).py                 # 主训练和测试脚本
├── model/
│   └── st_shift_gcn_fusion.py        # ST-Shift-GCN 融合模型定义
├── train/
│   └── train_model_fusion_2.py       # 训练器实现
├── test/
│   └── test_model_fusion_2.py        # 测试器实现
├── graph/                            # 图结构定义
│   ├── msr.py                       # MSR 数据集图结构
│   └── ut.py                        # UT 数据集图结构
├── dataset/                          # 数据集目录
└── README.md
```

## 🗂 数据集支持

| 数据集              | 节点数 | 动作类别数 | 图布局    |
| ------------------- | ------ | ---------- | --------- |
| MSRAction3D         | 20     | 20         | msr       |
| UTKinectAction3D    | 20     | 10         | ut        |
| Florence3DActions   | 15     | 9          | florence  |
| HanYueDailyAction3D | 25     | 15         | ntu-rgb+d |

## 🔧 模型架构

### ST-Shift-GCN 特点

- **空间建模**: 使用图卷积网络捕捉关节间的空间关系
- **时间建模**: 通过移位操作处理时间序列数据
- **多流融合**: 在 softmax 层融合多个数据流的概率输出
- **自适应图结构**: 支持不同数据集的专用图布局策略

### 融合策略

本项目采用 **softmax层概率融合** 方法，将不同流的预测概率进行加权融合，显著提升最终识别准确率。

## 📊 实验结果

训练过程会自动生成日志文件，格式为 `log_YYYY-MM-DD_HH_MM.txt`，包含：

- 模型配置和超参数
- 每个epoch的训练损失和准确率
- 测试集上的最终准确率
- 多流融合的详细结果

## 🤝 如何贡献

我们欢迎社区贡献！请参考以下步骤：

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送分支：`git push origin feature/AmazingFeature`
5. 提交 Pull Request

### 贡献指南

- 确保代码符合现有风格
- 添加适当的注释和文档
- 更新 README.md 以反映重大更改

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 创建 [Issue](https://github.com/HuangBen0209/DB-HPN/issues)
- 发送邮件至项目维护者

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者们。

---

