'''
Hypothesis: in-betweeness centrality computing from energy-neighbor model
'''
import numpy as np
from random_graph import *
import random
import time
import math


def generate_binary_representation(d, max_digits=8):
    return (((d[:, None] & (1 << np.arange(max_digits)))) > 0).astype(np.float32)


class RBM:

    def __init__(self, num_dimensions):
        self.dim = num_dimensions
        self.W = np.random.rand(self.dim, self.dim)
        self.seed = np.random.rand(self.dim, 64)

    def __str__(self):
        return str(self.W)

    def __forward(self, v):
        x = np.matmul(self.W, v)
        return 1 / (1 + np.exp(-x))

    def __backward(self, h):
        x = np.matmul(np.transpose(self.W), h)
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, v):
        return np.random.binomial(1, self.__forward(v))

    def sample_visible(self, h):
        return np.random.binomial(1, self.__backward(h))

    def persistance_mc(self, sampling_steps=20):
        for i in range(sampling_steps):
            self.seed = self.sample_visible(self.sample_hidden(self.seed))

    def incrementally_learn(self, v, lr=0.001):
        positive = np.matmul(self.sample_hidden(v), np.transpose(v))
        self.persistance_mc()
        negative = np.matmul(self.sample_hidden(self.seed), np.transpose(self.seed))
        self.W = self.W + lr * (positive - negative)

    def coordinate_ascend(self, v):
        return np.around(self.__backward(np.around(self.__forward(v))))

    def compute_prop(self, v):
        '''
        v has shape: [batch, dimension]
        '''
        all_h = generate_binary_representation(np.arange(math.pow(2, self.dim)).astype(np.int32) + 1, self.dim)
        partition = np.sum(np.matmul(all_h, np.matmul(self.W, np.transpose(all_h))))
        return np.sum(np.matmul(all_h, np.matmul(self.W, np.transpose(v))), axis=0) / partition


def build_energy_model(g, explore_steps=2000):
    size = g.shape[0]
    dimension = math.ceil(math.log(size, 2))
    bin_rep = generate_binary_representation(np.arange(size) + 1, dimension)
    model = RBM(dimension)
    for i in range(explore_steps):
        path = random_path(g, random.randint(0, g.shape[0] - 1), random.randint(0, g.shape[0] - 1))
        model.incrementally_learn(np.transpose(bin_rep[path]))
    return model


def count_in_betweeness_centrality(g, explore_steps=2000):
    stat = np.zeros([g.shape[0]])
    for i in range(explore_steps):
        path = random_path(g, random.randint(0, g.shape[0] - 1), random.randint(0, g.shape[0] - 1))
        stat[path] = stat[path] + 1

    return stat / np.sum(stat)


if __name__ == '__main__':
    # bin_rep = generate_binary_representation(np.arange(size) + 1, 8)
    # print(bin_rep)

    g = random_graph(128, 0.1)

    # directly compute
    print(count_in_betweeness_centrality(g))

    # using node degree
    out_going, in_coming = node_degrees(g)
    degrees = out_going + in_coming
    print(degrees / np.sum(degrees))

    # using energy model
    M = build_energy_model(g)
    bin_rep = generate_binary_representation(np.arange(g.shape[0]) + 1, 7)
    print(M.compute_prop(bin_rep))