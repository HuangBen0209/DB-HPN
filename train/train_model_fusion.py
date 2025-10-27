import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.cuda
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class STGCNTrainer:
    def __init__(self, model,train_data_path, train_label_path, num_nodes, use_residual, batch_size=32, num_epochs=300, lr=0.001,
                 patience=30, graph="graph.ntu_rgb_d.Graph",  # 默认图结构
                 graph_args=None, edge_importance_weighting=True, log_file_path="log.txt",
                 save_modelName="ST-GCN-ShiftGCN-fusion_model.pth"):

        self.batch_size = batch_size
        self.logfile_path = log_file_path
        self.graph = graph
        self.save_modelName = save_modelName

        if graph_args is None:
            self.graph_args = {'layout': 'openpose', 'strategy': 'adaptive'}
        else:
            self.graph_args = graph_args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.train_data = np.load(train_data_path)
        self.train_label = np.load(train_label_path)

        print(f"训练数据路径: {train_data_path}")  # 👈 打印训练数据路径
        print(f"训练数据形状: {self.train_data.shape}")  # 👈 打印训练数据维度
        print(f"训练标签形状: {self.train_label.shape}")  # 👈 打印标签维度

        self.train_data = torch.tensor(self.train_data, dtype=torch.float32).to(self.device)
        self.train_label = torch.tensor(self.train_label, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(self.train_data, self.train_label)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.num_classes = len(np.unique(self.train_label.cpu().numpy()))
        self.in_channels = self.train_data.shape[1]  # 👈 动态获取输入通道
        self.graph_args = graph_args
        self.edge_importance_weighting = edge_importance_weighting
        self.num_nodes = num_nodes
        self.use_residual = use_residual

        # 只初始化模型，如果传入的是类
        if isinstance(model, nn.Module):
            self.model = model.to(self.device)
        else:
            self.model = model(
                num_class=self.num_classes,
                num_nodes=self.num_nodes,
                num_person=1,
                graph=self.graph,
                graph_args=self.graph_args,
                in_channels=self.in_channels,
                edge_importance_weighting=self.edge_importance_weighting,
                use_residual=self.use_residual
            ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.patience = patience
        self.best_epoch = 0
        self.counter = 0
        self.best_train_loss = float('inf')
        self.num_epochs = num_epochs

    def calculate_accuracy(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    def train(self):
        with open( self.logfile_path, 'a') as log_file:
            previous_lr = self.optimizer.param_groups[0]['lr']
            print(f"Initial learning rate: {previous_lr}", file=log_file, flush=True)
            for epoch in range(self.num_epochs):
                self.model.train()
                running_loss = 0.0
                for i, (inputs, labels) in enumerate(self.train_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                train_loss = running_loss / len(self.train_loader)
                self.scheduler.step(train_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                # if current_lr != previous_lr:
                #     print(f'Learning rate decreased from {previous_lr} to {current_lr}', file=log_file, flush=True)
                #     previous_lr = current_lr
                train_accuracy = self.calculate_accuracy(self.train_loader)
                log_msg = f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%'
                print(log_msg)

                if train_loss < self.best_train_loss:
                    self.best_train_loss = train_loss
                    self.best_epoch = epoch
                    self.best_train_accuracy = train_accuracy  # 记录最优 epoch 的准确率
                    self.counter = 0
                    print(f'Saving_{self.save_modelName}')
                    torch.save(self.model.state_dict(), self.save_modelName)
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        stop_msg = f'early stopping {epoch + 1}.best epoch is{self.best_epoch + 1}.'
                        print(f"{stop_msg}")
                        break
            # 写入最优 epoch 的信息
            # best_msg = f'\t best epoch: {self.best_epoch + 1}, Loss: {self.best_train_loss:.4f}, Accuracy: {self.best_train_accuracy * 100:.2f}%'
            # print(f"{best_msg}"+'\t', file=log_file, flush=True)

