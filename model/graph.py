import numpy as np
import torch
import torch.nn as nn


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf  # 最大跳数为无穷大
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]  # 计算A的d次幂，表示i到j的d步距离为d
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):  # 归一化有向图
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):  # 归一化无向图
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class Graph:
    def __init__(self, layout='ntu-rgb+d', strategy='uniform', max_hop=1, dilation=1, num_nodes=25):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = num_nodes

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):
        if layout == 'openpose':
            self_link = [(i, i) for i in range(self.num_node)]
            # neighbor_link = [(0, 1), (1, 2), (1, 3), (2, 4), (1, 5), (3, 6), (4, 7), (5, 8), (6, 10), (5, 9), (9, 11),
            #                  (10, 12), (12, 13), (13, 14)]
            neighbor_link = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
                             (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                             (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
                             (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
                             (21, 22), (22, 7), (23, 24), (24, 11)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (21, 3), (3, 4),  # 躯干
                              (21, 5), (5, 6), (6, 7), (7, 8),  # 左臂
                              (21, 9), (9, 10), (10, 11), (11, 12),  # 右臂
                              (1, 13), (13, 14), (14, 15), (16, 15),  # 左腿
                              (1, 17), (17, 18), (18, 19), (19, 20),  # 右腿
                              (8, 22), (7, 23), (12, 24), (11, 25)]  # 细小关节
            # ,(22, 23), (23, 8), (24, 25), (25, 12)] 官方细小关节的连接方式
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'florence':
            self.num_node = 15
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = \
                [(0, 1), (1, 2),  # 躯干
                 (1, 3), (3, 4), (4, 5),  # 左臂
                 (1, 6), (6, 7), (7, 8),  # 右臂
                 (0, 9), (9, 10), (10, 11),  # 左腿
                 (0, 12), (12, 13), (13, 14)]  # 右腿

            #    [(0, 1), (1, 2), (1, 3), (2, 4), (1, 5), (3, 6), (4, 7), (5, 8), (6, 10), (5, 9), (9, 11),
            # (10, 12), (12, 13), (13, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'msr':
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = \
                [
                    [1, 2],  # SpineBase - SpineMid
                    [2, 20],  # SpineMid - SpineShoulder
                    [20, 3],  # SpineShoulder - Head

                    [20, 4],  # SpineShoulder - ShoulderLeft
                    [20, 8],  # SpineShoulder - ShoulderRight
                    [4, 5],  # ShoulderLeft - ElbowLeft
                    [5, 6],  # ElbowLeft - WristLeft
                    [6, 7],  # WristLeft - HandLeft
                    [8, 9],  # ShoulderRight - ElbowRight
                    [9, 10],  # ElbowRight - WristRight
                    [10, 11],  # WristRight - HandRight
                    [1, 12],  # SpineBase - HipLeft
                    [12, 13],  # HipLeft - KneeLeft
                    [13, 14],  # KneeLeft - AnkleLeft
                    [14, 15],  # AnkleLeft - FootLeft
                    [1, 16],  # SpineBase - HipRight
                    [16, 17],  # HipRight - KneeRight
                    [17, 18],  # KneeRight - AnkleRight
                    [18, 19]  # AnkleRight - FootRight
                ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'ut':
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = \
                [
                    [1, 2],  # SpineBase - SpineMid
                    [2, 3],  # SpineMid - neck
                    [3, 4],  # neck - Head

                    [3, 5],  # neck - ShoulderLeft
                    [3, 9],  # neck - ShoulderRight

                    [5, 6],  # ShoulderLeft - ElbowLeft
                    [6, 7],  # ElbowLeft - WristLeft
                    [7, 8],  # WristLeft - HandLeft
                    [9, 10],  # ShoulderRight - ElbowRight
                    [10, 11],  # ElbowRight - WristRight
                    [11, 12],  # WristRight - HandRight
                    [1, 13],  # SpineBase - HipLeft
                    [13, 14],  # HipLeft - KneeLeft
                    [14, 15],  # KneeLeft - AnkleLeft
                    [15, 16],  # AnkleLeft - FootLeft
                    [1, 17],  # SpineBase - HipRight
                    [17, 18],  # HipRight - KneeRight
                    [18, 19],  # KneeRight - AnkleRight
                    [19, 20]  # AnkleRight - FootRight
                ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'customer':
            self_link = [(i, i) for i in range(self.num_node)]
            # neighbor_link = [(0, 1), (1, 2), (1, 3), (2, 4), (1, 5), (3, 6), (4, 7), (5, 8), (6, 10), (5, 9), (9, 11),
            #                  (10, 12), (12, 13), (13, 14)]
            neighbor_link = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
                             (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                             (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
                             (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
                             (21, 22), (22, 7), (23, 24), (24, 11)]
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_undigraph(adjacency)

        if strategy == 'uniform':  # 自连接和手工邻接矩阵
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'three_streams':  # 三流邻接矩阵：自连接、中心节点->外围节点、外围节点->中心节点
            self_loop = np.eye(self.num_node)
            centripetal = np.zeros((self.num_node, self.num_node))
            centrifugal = np.zeros((self.num_node, self.num_node))

            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[i, j] == 1:
                        if i == self.center:
                            centrifugal[i, j] = 1
                        elif j == self.center:
                            centripetal[i, j] = 1

            self_loop = normalize_digraph(self_loop)
            centripetal = normalize_digraph(centripetal)
            centrifugal = normalize_digraph(centrifugal)

            A = np.stack([self_loop, centripetal, centrifugal])
            self.A = A
        elif strategy == 'adaptive':  # 自适应邻接矩阵
            A = torch.tensor(normalize_adjacency, dtype=torch.float32)
            A = A.unsqueeze(0)
            self.A = nn.Parameter(A, requires_grad=True)
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':  # 根据跳数分离邻接矩阵：根部流（root）：如果节点j和节点i到中心节点的跳数相同。
            # 靠近中心的流（close）：如果节点j比节点i更远离中心节点。
            # 远离中心的流（further）：如果节点j比节点i更靠近中心节点。
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")
