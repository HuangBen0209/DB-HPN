import numpy as np
import torch


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


def edge2mat(link, num_node): # 将边列表转换为邻接矩阵
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 归一化有向图
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
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

def get_uniform_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_adaptive_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    A = torch.tensor(A, dtype=torch.float32)
    A = torch.nn.Parameter(A, requires_grad=True)
    return A

def get_adjacency(strategy='uniform',max_hop=1, dilation=1,num_node=25, self_link=None, inward=None, outward=None):
    if strategy == 'uniform':
        A = get_uniform_graph(num_node, self_link, inward, outward)
    if strategy == 'spatial':
        A = get_spatial_graph(num_node, self_link, inward, outward)
    if strategy == 'adaptive':
        A = get_adaptive_graph(num_node, self_link, inward, outward)
    return A

if __name__ == "__main__":
    num_node = 10

    # 自连接边，每个节点和自己连接
    self_link = [(i, i) for i in range(num_node)]

    # 入向边（假设9个节点，手动设计一条简单链路）
    inward = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7),(3,5)]

    # 出向边是入向边的反向
    outward = [(j, i) for (i, j) in inward]

    A = get_spatial_graph(num_node, self_link, inward, outward)

    print("邻接矩阵张量A形状:", A.shape)  # (3, 9, 9)
    print("\n自连接矩阵I:")
    print(A[0])
    print("\n入向归一化矩阵In:")
    print(A[1])
    print("\n出向归一化矩阵Out:")
    print(A[2])
