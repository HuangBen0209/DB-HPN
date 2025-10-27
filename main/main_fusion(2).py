# -*- coding: utf-8 -*-
# 模型：st-shift-gcn-fusion（2）.py  在softmax层进行概率融合
import os
import random
from datetime import datetime

import numpy as np
import torch

from model.st_shift_gcn_fusion import Model as st_shiftgcn
from test.test_model_fusion_2 import STGCNTester
from train.train_model_fusion_2 import STGCNTrainer


def set_seed(seed):
    random.seed(seed)  # 设置 Python 内置随机模块的种子，影响 random.random() 等
    np.random.seed(seed)  # 设置 NumPy 的随机数种子，影响 np.random 模块的所有函数
    torch.manual_seed(seed)  # 设置 PyTorch 的 CPU 随机数种子，影响 torch.rand()、torch.randn() 等
    torch.cuda.manual_seed(seed)  # 设置当前 GPU 的随机数种子（如果使用 CUDA）
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子（多 GPU 训练时使用）
    torch.backends.cudnn.deterministic = True  # 设置 cuDNN 的确定性模式，保证每次运行结果一致（会降低训练速度）
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动优化算法搜索，以保证结果可复现（否则每次可能选不同算法）


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(__file__)
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

    # # 定义joint流的数据集路径
    # joint_dataset_paths = {
    #     "Florence_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_origin/joint"),
    #     # "Florence_filtered_len_s": os.path.join(BASE_DIR, "dataset/Florence3DActions_filtered_len_s/joint"),
    #     # "Florence_norm23_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm23_origin/joint"),
    #     # "Florence_norm23_filtered_len_s": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm23_filtered_len_s/joint"),
    #     # "MSR_origin": os.path.join(BASE_DIR, "dataset/MSRAction3D_origin/joint"),
    #     # "MSR_filtered_len_s": os.path.join(BASE_DIR, "dataset/MSRAction3D_filtered_len_s/joint"),
    #     # "MSR_norm60_origin": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm60_origin/joint"),
    #     # "MSR_norm60_filtered_len_s": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm60_filtered_len_s/joint"),
    #     # "UT_origin": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_origin/joint"),
    #     # "UT_filtered_len_s": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_filtered_len_s/joint"),
    #     # "UT_norm60_origin": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm60_origin/joint"),
    #     # "UT_norm60_filtered_len_s": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm60_filtered_len_s/joint"),
    #     # "HanYue_origin": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_origin/joint"),
    #     # "HanYue_filtered_len_s": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_filtered_len_s/joint"),
    #     # "HanYue_norm70_origin": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_norm70_origin/joint"),
    #     # "HanYue_norm70_filtered_len_s": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_norm70_filtered_len_s/joint"),
    # }

    # 定义bone流的数据集路径（根据你的实际路径修改）
    bone_dataset_paths = {
        # "Florence_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_origin/bone"),
        # "Florence_filtered_len_s": os.path.join(BASE_DIR, "dataset/Florence3DActions_filtered_len_s/bone"),
        # "Florence_norm23_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm23_origin/bone"),
        # "Florence_norm23_filtered_len_s": os.path.join(BASE_DIR,
        #                                                "dataset/Florence3DActions_norm23_filtered_len_s/bone"),
        "MSR_origin": os.path.join(BASE_DIR, "dataset/MSRAction3D_origin/joint"),
        "MSR_filtered_len_s": os.path.join(BASE_DIR, "dataset/MSRAction3D_filtered_len_s/joint"),
        "MSR_norm60_origin": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm60_origin/joint"),
        "MSR_norm60_filtered_len_s": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm60_filtered_len_s/joint"),

        "UT_origin": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_origin/joint"),
        "UT_filtered_len_s": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_filtered_len_s/joint"),
        "UT_norm60_origin": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm60_origin/joint"),
        "UT_norm60_filtered_len_s": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm60_filtered_len_s/joint"),

        # "HanYue_origin": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_origin/bone"),
        # "HanYue_filtered_len_s": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_filtered_len_s/bone"),
        # "HanYue_norm70_origin": os.path.join(BASE_DIR, "da       taset/HanYueDailyAction3D_norm70_origin/bone"),
        # "HanYue_norm70_filtered_len_s": os.path.join(BASE_DIR,
        #                                              "dataset/HanYueDailyAction3D_norm70_filtered_len_s/bone"),
    }
    # 配置映射：数据集组名 => 配置
    dataset_group_configs = {
        "HanYue": {
            "num_nodes": 25,
            "num_classes": 15,
            "in_channels": 3,
            "graph": "graph.ntu_rgb_d.Graph",
            "graph_args_list": [{'layout': 'ntu-rgb+d', 'strategy': 'adaptive'}],
        },
        "Florence": {
            "num_nodes": 15,
            "num_classes": 9,
            "in_channels": 3,
            "graph": "graph.florence.Graph",
            "graph_args_list": [{'layout': 'florence', 'strategy': 'adaptive'}],
        },
        "MSR": {
            "num_nodes": 20,
            "num_classes": 20,
            "in_channels": 3,
            "graph": "graph.msr.Graph",
            "graph_args_list": [{'layout': 'msr', 'strategy': 'adaptive'}],
        },
        "UT": {
            "num_nodes": 20,
            "num_classes": 10,
            "in_channels": 3,
            "graph": "graph.ut.Graph",
            "graph_args_list": [{'layout': 'ut', 'strategy': 'adaptive'}],
        }
    }
    # 训练参数
    timestamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
    log_path = os.path.join(CURRENT_DIR, f"log_{timestamp}.txt")

    use_residual = True
    lr = 0.001
    patience = 30
    edge_importance_weighting = True
    num_epochs = 300
    run_seeds = [1]  # 用于不同运行的随机种子
    batch_size_list = [16]

    # num_epochs = 0
    # run_seeds = [1]  # 用于不同运行的随机种子
    # batch_size_list = [8]

    use_model = st_shiftgcn  # 使用的模型类型: st_gcn, st_shiftgcn ,shiftgcn
    save_model_name = f"model_st_shiftgcn.pth"
    save_model_name = f"model_st_gcn.pth"
    # 训练
    for dataset_name, data_dirs in bone_dataset_paths.items():
        group_key = dataset_name.split("_")[0]  # 提取前缀，如 "HanYue"
        if group_key not in dataset_group_configs:
            raise ValueError(f"Unsupported dataset group: {group_key}")
        config = dataset_group_configs[group_key]
        num_nodes = config["num_nodes"]
        num_classes = config["num_classes"]
        graph = config["graph"]
        graph_args_list = config["graph_args_list"]
        in_channels = config["in_channels"]
        with open(log_path, 'a', encoding='utf-8') as log_file:
            for graph_args in graph_args_list:
                for batch_size in batch_size_list:
                    total_acc = 0.0
                    header = f"""
                    softmax层进行概率融合
================================================================================
                Model         : {save_model_name}
                Dataset       : {dataset_name}
                Graph layout  : {graph_args['layout']}
                Graph strategy: {graph_args['strategy']}
                Batch size    : {batch_size}
                Use residual  : {use_residual}
                Learning rate : {lr}
                Epochs        : {num_epochs}
================================================================================
           """
                    print(header.strip(), file=log_file, flush=True)
                    for i, seed in enumerate(run_seeds):
                        set_seed(seed)
                        print(f"Run {i + 1}/{len(run_seeds)} | Seed = {seed}", file=log_file, flush=True)
                        # 1. 原始数据训练和测试
                        train_data_path = os.path.join(data_dirs, "train_data.npy")
                        train_label_path = os.path.join(data_dirs, "train_label.npy")
                        test_data_path = os.path.join(data_dirs, "test_data.npy")
                        test_label_path = os.path.join(data_dirs, "test_label.npy")
                        model_st_shiftgcn = st_shiftgcn(
                            num_class=num_classes,
                            num_nodes=num_nodes,
                            num_person=1,
                            graph=graph,
                            in_channels=in_channels,
                        )
                        trainer = STGCNTrainer(model_st_shiftgcn, train_data_path, train_label_path,
                                               num_nodes,
                                               use_residual,
                                               batch_size,
                                               num_epochs, lr, patience,
                                               graph, graph_args, edge_importance_weighting,
                                               log_file_path=log_path,
                                               save_modelName=save_model_name
                                               )  # 保存第二个模型，
                        trainer.train()
                        tester = STGCNTester(test_data_path, test_label_path, num_nodes, batch_size,
                                             graph, graph_args,
                                             edge_importance_weighting, use_residual=use_residual,
                                             log_file_path=log_path,
                                             )  # 加载第一个模型测试
                        tester.load_model(save_model_name)
                        st_acc, shift_acc, fusion_acc_ = tester.test()
                        # print(f"  \t原始数据准确率: {test_acc_origin:.2f}%", file=log_file, flush=True)
                        total_acc += fusion_acc_

                    avg_acc = total_acc / len(run_seeds)
                    print(f"\n>>> 平均准确率（滤波数据，{len(run_seeds)}次重复）: {avg_acc:.2f}%\n\n", file=log_file,
                          flush=True)
