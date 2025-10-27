import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model.shift_gcn import Model as shift_gcn
from model.st_gcn import STGCN as st_gcn
from model.st_shift_gcn import Model as st_shiftgcn

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
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        log_msg = f' \t Test Accuracy is: {accuracy:.2f}%'
        print(log_msg)

        return accuracy
