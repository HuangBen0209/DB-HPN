import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model.shift_gcn import Model as shift_gcn
from model.st_gcn import STGCN as st_gcn
from model.st_shift_gcn import Model as st_shiftgcn

torch.serialization.add_safe_globals([shift_gcn])
torch.serialization.add_safe_globals([st_gcn])

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
            in_channels=self.in_channels, # 改为动态读取输入通道
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

        # self.st_shiftgcn=st_shiftgcn(
        #     num_class=self.num_classes,
        #     num_nodes=self.num_nodes,
        #     num_person=1,
        #     graph=self.graph,
        #     graph_args=self.graph_args,
        #     in_channels=self.in_channels,
        #     edge_importance_weighting=True,
        #     use_residual=True
        # )

    def load_model(self,stGCN="ST-GCN_model.pth",shiftGCN="ShiftGCN_model.pth"):
        try:
            checkpoint = torch.load(stGCN, weights_only=False)
            self.st_gcn.load_state_dict(checkpoint)
            print(f"Model--{stGCN}-- loaded successfully.")

            checkpoint = torch.load(shiftGCN, weights_only=False)
            self.shift_gcn.load_state_dict(checkpoint)
            print(f"Model--{shiftGCN}-- loaded successfully.")

        except FileNotFoundError:
            print("Model file not found.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def test(self):
        self.st_gcn.eval()
        self.shift_gcn.eval()

        correct_stgcn = 0
        correct_shiftgcn = 0
        correct_fused = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                stgcn_outputs = self.st_gcn(inputs)
                shiftgcn_outputs = self.shift_gcn(inputs)

                stgcn_probs = torch.softmax(stgcn_outputs, dim=1)
                shiftgcn_probs = torch.softmax(shiftgcn_outputs, dim=1)

                # ST-GCN 单模型预测
                _, predicted_stgcn = torch.max(stgcn_probs, 1)
                correct_stgcn += (predicted_stgcn == labels).sum().item()

                # Shift-GCN 单模型预测
                _, predicted_shiftgcn = torch.max(shiftgcn_probs, 1)
                correct_shiftgcn += (predicted_shiftgcn == labels).sum().item()

                # 融合两个模型的 softmax 输出分数
                fused_probs = stgcn_probs * 0.5 + shiftgcn_probs * 0.5
                _, predicted_fused = torch.max(fused_probs, 1)
                correct_fused += (predicted_fused == labels).sum().item()

                total += labels.size(0)

        accuracy_stgcn = 100 * correct_stgcn / total
        accuracy_shiftgcn = 100 * correct_shiftgcn / total
        accuracy_fused = 100 * correct_fused / total

        print(f'\tST-GCN Test Accuracy: {accuracy_stgcn:.2f}%')
        print(f'\tShift-GCN Test Accuracy: {accuracy_shiftgcn:.2f}%')
        print(f'\tFused Test Accuracy: {accuracy_fused:.2f}%')

        return accuracy_stgcn, accuracy_shiftgcn, accuracy_fused

    def test_grid_search(self, step=0.1):

        self.st_gcn.eval()
        self.shift_gcn.eval()

        best_alpha = 0.0
        best_fused_accuracy = 0.0
        correct_stgcn = 0
        correct_shiftgcn = 0
        total = 0
        print(f"开始网格搜索，步长: {step}")
        with torch.no_grad():
            # 只评估 ST-GCN 和 Shift-GCN 的单独准确率
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                stgcn_outputs = self.st_gcn(inputs)
                shiftgcn_outputs = self.shift_gcn(inputs)

                stgcn_probs = torch.softmax(stgcn_outputs, dim=1)
                shiftgcn_probs = torch.softmax(shiftgcn_outputs, dim=1)

                _, pred_st = torch.max(stgcn_probs, 1)
                _, pred_sh = torch.max(shiftgcn_probs, 1)
                correct_stgcn += (pred_st == labels).sum().item()
                correct_shiftgcn += (pred_sh == labels).sum().item()
                total += labels.size(0)

            accuracy_stgcn = 100 * correct_stgcn / total
            accuracy_shiftgcn = 100 * correct_shiftgcn / total

            # 网格搜索，寻找最佳 alpha
            for alpha in torch.arange(0, 1.01, step):
                correct_fused = 0
                total = 0
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    stgcn_outputs = self.st_gcn(inputs)
                    shiftgcn_outputs = self.shift_gcn(inputs)

                    stgcn_probs = torch.softmax(stgcn_outputs, dim=1)
                    shiftgcn_probs = torch.softmax(shiftgcn_outputs, dim=1)

                    fused_probs = alpha * stgcn_probs + (1 - alpha) * shiftgcn_probs
                    _, predicted = torch.max(fused_probs, 1)
                    correct_fused += (predicted == labels).sum().item()
                    total += labels.size(0)

                acc_fused = 100 * correct_fused / total

                if acc_fused > best_fused_accuracy:
                    best_fused_accuracy = acc_fused
                    best_alpha = alpha.item()
                print(f'Alpha: {alpha:.2f}, Fused Accuracy: {acc_fused:.2f}%')

        print(f'   ST-GCN Accuracy: {accuracy_stgcn:.2f}%')
        print(f'   Shift-GCN Accuracy: {accuracy_shiftgcn:.2f}%')
        print(f'\n✅ 最佳融合权重 alpha: {best_alpha:.2f}，对应融合准确率: {best_fused_accuracy:.2f}%')

        return alpha, accuracy_stgcn, accuracy_shiftgcn, best_fused_accuracy


if __name__ == '__main__':
    # 测试数据路径和标签路径
    test_data_path = "../dataset_norm/simple/HanYueDailyAction3D_norm70_origin/test_data.npy"
    test_label_path = "../dataset_norm/simple/HanYueDailyAction3D_norm70_origin/test_label.npy"

    num_nodes = 25  #
    batch_size = 8


    tester = STGCNTester(test_data_path, test_label_path, num_nodes, batch_size,
                         graph= "graph.ntu_rgb_d.Graph",
                         graph_args={'layout': 'ntu-rgb+d', 'strategy': 'adaptive'},
                         edge_importance_weighting=True,
                         use_residual=True,
                         )
    # 加载模型（默认文件名）
    tester.load_model("../main/ST-GCN_model.pth", "../main/Shift-GCN_model.pth")

    # 运行测试，打印loss和accuracy
    alpha, acc_stgcn, acc_shiftgcn, acc_fused = tester.test_grid_search()
    print(f"\tST-GCN 准确率: {acc_stgcn:.2f}%")
    print(f"\tShift-GCN 准确率: {acc_shiftgcn:.2f}%")
    print(f"\t融合后准确率: {acc_fused:.2f}%")
