import sys

sys.path.extend(['../'])
from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward_ori_index =  [
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
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
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

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
