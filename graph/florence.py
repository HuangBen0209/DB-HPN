import sys
sys.path.extend(['../'])
from graph import tools

num_node = 15
self_link = [(i, i) for i in range(num_node)]

inward1base =  neighbor_link = [(1, 2), (2, 3), (2, 4), (4, 5),
                                (5, 6), (2, 7), (7, 8), (8, 9),
                                (1, 10), (10, 11), (11, 12),
                                (1, 13), (13, 14), (14, 15)]
inward = [(i - 1, j - 1) for (i, j) in inward1base]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode='spatial'):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
