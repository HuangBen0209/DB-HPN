import os
import random
from datetime import datetime

import numpy as np
import torch

from model.shift_gcn import Model as shiftgcn
from model.st_gcn import STGCN as st_gcn
from model.st_shift_gcn import Model as st_shiftgcn
from test.test_model import STGCNTester
from train.train_model import STGCNTrainer


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

    # # 所有原始数据集路径
    # dataset_paths = {
    #
    #     # "Florence_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_origin"),
    #     # "Florence_filtered_len_s": os.path.join(BASE_DIR, "dataset/Florence3DActions_filtered_len_s"),
    #     #
    #     # "Florence_norm20_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm20_origin"),
    #     # "Florence_norm20_filtered_len_s": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm20_filtered_len_s"),
    #     #
    #     # "Florence_norm23_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm23_origin"),
    #     # "Florence_norm23_filtered_len_s": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm23_filtered_len_s"),
    #     #
    #     # "MSR_origin": os.path.join(BASE_DIR, "dataset/MSRAction3D_origin"),
    #     # "MSR_filtered_len_s": os.path.join(BASE_DIR, "dataset/MSRAction3D_filtered_len_s"),
    #     # "MSR_norm60_origin": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm60_origin"),
    #     # "MSR_norm60_filtered_len_s": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm60_filtered_len_s"),
    #     #
    #     # "UT_origin": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_origin"),
    #     # "UT_filtered_len_s": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_filtered_len_s"),
    #     # "UT_norm60_origin": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm60_origin"),
    #     # "UT_norm60_filtered_len_s": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm60_filtered_len_s"),
    #     #
    #     # "HanYue_origin": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_origin"),
    #     # "HanYue_filtered_len_s": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_filtered_len_s"),
    #     # "HanYue_norm70_origin": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_norm70_origin"),
    #     # "HanYue_norm70_filtered_len_s": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_norm70_filtered_len_s"),
    #     "HanYue_norm100_origin ": os.path.join(BASE_DIR, "dataset/HanYue_norm100_origin"),
    # }

    dataset_paths = {

        "Florence_norm23_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm23_origin/joint"),
        # "Florence_norm30_origin": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm30_origin"),
        #
        # "MSR_norm40_origin": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm40_origin"),
        # "MSR_norm60_origin": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm60_origin"),
        #
        # "UT_norm40_origin": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm40_origin"),
        # "UT_norm60_origin": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm60_origin"),
        #
        # "HanYue_norm70_origin": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_norm70_origin"),
        # "HanYue_norm100_origin": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_norm100_origin"),
    }
    #测试参数量用的数据集
    # dataset_paths = {
    #     "Florence_23": os.path.join(BASE_DIR, "dataset/Florence3DActions_norm23_origin"),
    #     "MSR_60": os.path.join(BASE_DIR, "dataset/MSRAction3D_norm60_origin"),
    #     "UT_60": os.path.join(BASE_DIR, "dataset/UTKinectAction3D_norm60_origin"),
    #     "HanYue_70": os.path.join(BASE_DIR, "dataset/HanYueDailyAction3D_norm70_origin"),
    # }

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
    run_seeds = [1, 42, 666, 3407, 114514]  # 用于不同运行的随机种子
    run_seeds = [1]  # 用于不同运行的随机种子

    batch_size_list = [16]

    use_model = shiftgcn  # 使用的模型类型: st_gcn, st_shiftgcn ,shiftgcn
    save_model_name = f"model_shiftgcn.pth"

    # 训练
    for dataset_name, data_dirs in dataset_paths.items():
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

                        model_stgcn = st_gcn(
                            num_class=num_classes,
                            num_nodes=num_nodes,
                            graph_args=graph_args,
                            in_channels=in_channels,
                            edge_importance_weighting=True,
                            use_residual=True
                        )
                        model_shiftgcn = shiftgcn(
                            num_class=num_classes,
                            num_nodes=num_nodes,
                            num_person=1,
                            graph=graph,
                            in_channels=in_channels,
                        )
                        model_st_shiftgcn = st_shiftgcn(
                            num_class=num_classes,
                            num_nodes=num_nodes,
                            num_person=1,
                            graph=graph,
                            # graph_args=graph_args,
                            in_channels=in_channels,
                            # edge_importance_weighting=True,
                            # use_residual=True
                        )
                        if (use_model == st_gcn):
                            trainer = STGCNTrainer(model_stgcn, train_data_path, train_label_path, num_nodes,
                                                   use_residual,
                                                   batch_size,
                                                   num_epochs, lr, patience,
                                                   graph, graph_args, edge_importance_weighting,
                                                   log_file_path=log_path,
                                                   save_modelName=save_model_name
                                                   )  # 保存第一个模型，
                        elif (use_model == shiftgcn):
                            trainer = STGCNTrainer(model_shiftgcn, train_data_path, train_label_path,
                                                   num_nodes,
                                                   use_residual,
                                                   batch_size,
                                                   num_epochs, lr, patience,
                                                   graph, graph_args, edge_importance_weighting,
                                                   log_file_path=log_path,
                                                   save_modelName=save_model_name
                                                   )  # 保存第二个模型，
                        elif (use_model == st_shiftgcn):
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
                        test_acc_origin = tester.test()
                        print(f"  \t原始数据准确率: {test_acc_origin:.2f}%", file=log_file, flush=True)

                        total_acc += test_acc_origin

                    avg_acc = total_acc / len(run_seeds)
                    print(f"\n>>> 平均准确率（滤波数据，{len(run_seeds)}次重复）: {avg_acc:.2f}%\n\n", file=log_file,
                          flush=True)
