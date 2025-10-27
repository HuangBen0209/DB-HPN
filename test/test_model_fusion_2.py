# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model.shift_gcn import Model as shift_gcn
from model.st_gcn import STGCN as st_gcn
from model.st_shift_gcn_fusion import Model as st_shiftgcn

torch.serialization.add_safe_globals([shift_gcn])
torch.serialization.add_safe_globals([st_gcn])
torch.serialization.add_safe_globals([st_shiftgcn])

class STGCNTester:
    def __init__(self, test_data_path, test_label_path, num_nodes, batch_size=32,
                 graph="graph.ntu_rgb_d.Graph",  # 默认图结构
                 graph_args=None, edge_importance_weighting=True, use_residual=True, log_file_path="log.txt",
                ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logfile_path = log_file_path
        self.graph = graph
        print(f"Using device: {self.device}")

        self.test_data = np.load(test_data_path)
        self.test_label = np.load(test_label_path)

        print(f"测试数据路径: {test_data_path}")  # <<== 打印测试数据路径
        print(f"测试数据形状: {self.test_data.shape}")  # <<== 打印测试数据形状
        print(f"测试标签形状: {self.test_label.shape}")  # <<== 打印测试标签形状

        self.test_data = torch.tensor(self.test_data, dtype=torch.float32).to(self.device)
        self.test_label = torch.tensor(self.test_label, dtype=torch.long).to(self.device)

        test_dataset = TensorDataset(self.test_data, self.test_label)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.num_classes = len(np.unique(self.test_label.cpu().numpy()))
        self.in_channels = self.test_data.shape[1]
        self.graph_args = graph_args
        self.num_nodes = num_nodes
        self.edge_importance_weighting = edge_importance_weighting
        self.use_residual = use_residual

        self.st_gcn = st_gcn(
            num_class=self.num_classes,
            num_nodes=self.num_nodes,
            graph_args=self.graph_args,
            in_channels=self.in_channels,  # 改为动态读取输入通道
            edge_importance_weighting=self.edge_importance_weighting,
            use_residual=self.use_residual
        ).to(self.device)

        self.shift_gcn = shift_gcn(
            num_class=self.num_classes,
            num_nodes=self.num_nodes,
            num_person=1,
            graph=self.graph,
            in_channels=self.in_channels  # 改为动态读取输入通道
        ).to(self.device)

        self.st_shiftgcn = st_shiftgcn(
            num_class=self.num_classes,
            num_nodes=self.num_nodes,
            num_person=1,
            graph=self.graph,
            in_channels=self.in_channels,
            # use_residual=True
        ).to(self.device)
        self.model= None  # 初始化模型为None
    def load_model(self,load_modelName="model_stgcn.pth"):
        try:
            if(load_modelName=="model_st_gcn.pth"):
                self.model = self.st_gcn
            elif(load_modelName=="model_shiftgcn.pth"):
                self.model = self.shift_gcn
            elif(load_modelName=="model_st_shiftgcn.pth"):
                self.model = self.st_shiftgcn
            checkpoint = torch.load(load_modelName, weights_only=False)
            self.model.load_state_dict(checkpoint)
            print(f"Model--{load_modelName}-- loaded successfully.")
        except FileNotFoundError:
            print("Model file not found.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def test(self):
        self.model.eval()
        for i in range(3):
            if i == 0:
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in self.test_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        st_outputs,_,_ ,weight = self.model(inputs)
                        _, predicted = torch.max(st_outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    st_accuracy = 100 * correct / total
            elif i == 1:
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in self.test_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        _, shift_outputs, _, weight = self.model(inputs)
                        _, predicted = torch.max(shift_outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    shift_accuracy = 100 * correct / total
            elif i == 2:
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in self.test_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        _, _, fusion_outputs, weight = self.model(inputs)
                        _, predicted = torch.max(fusion_outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    print(f"weight: {weight}")  # 打印权重
                    fusion_acc = 100 * correct / total
        log_msg = f'st_acc:{st_accuracy:.3f}%;\t shift_acc:{shift_accuracy:.3f}%;\t fusion_Accuracy is: {fusion_acc:.2f}%;\t fusion weight: {weight.cpu().numpy()}'
        print(log_msg)
        # 写入日志文件
        with open( self.logfile_path, 'a') as f:
            print(log_msg, file=f, flush=True)
        return st_accuracy,shift_accuracy,fusion_acc
if __name__ == '__main__':
    # 你的测试数据路径和标签s路径
    test_data_path = r"D:\MyCode\ActionRecognitionAll\shift-gcn-hb2\dataset\Florence3DActions_origin\test_data.npy"
    test_label_path = r"D:\MyCode\ActionRecognitionAll\shift-gcn-hb2\dataset\Florence3DActions_origin\test_label.npy"

    num_nodes = 15  #
    batch_size = 8

    # 初始化测试器
    tester = STGCNTester(
        test_data_path=test_data_path,
        test_label_path=test_label_path,
        num_nodes=num_nodes,
        batch_size=batch_size,
        graph="graph.florence.Graph"
    )
    # 加载模型（默认文件名）
    tester.load_model("D:\MyCode\ActionRecognitionAll\shift-gcn-hb2\main\ShiftGCN_filtered_len_model.pth")

    # 运行测试，打印loss和accuracy
    acc = tester.test()
    print(f"测试完成，Accuracy: {acc:.2f}%")
