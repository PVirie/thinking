'''
Hypothesis: in-betweeness centrality computing from energy-neighbor model
'''
import numpy as np
from random_graph import *
import random
import time
import math
import energy
import matplotlib.pyplot as plt


def generate_binary_representation(d, max_digits=8):
    return (((d[:, None] & (1 << np.arange(max_digits)))) > 0).astype(np.float32)


def generate_onehot_representation(d, max_digits=8):
    b = np.zeros((d.size, max_digits))
    b[np.arange(d.size), d] = 1
    return b


class RBM:

    def __init__(self, num_dimensions):
        self.dim = num_dimensions
        self.W = np.random.normal(0, 0.001, [self.dim, self.dim])
        self.seed = self.__backward(np.random.rand(self.dim, 64))

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
            self.seed = self.__backward(self.sample_hidden(self.seed))

    def incrementally_learn(self, v, lr=0.01):
        positive = np.matmul(self.__forward(v), np.transpose(v)) / v.shape[1]
        self.persistance_mc()
        negative = np.matmul(self.__forward(self.seed), np.transpose(self.seed)) / self.seed.shape[1]
        self.W = self.W + lr * (positive - negative)

    def coordinate_ascend(self, v):
        return np.around(self.__backward(np.around(self.__forward(v))))

    def compute_prop(self, v):
        '''
        v has shape: [dimension, batch]
        '''
        all_h = generate_binary_representation(np.arange(math.pow(2, self.dim)).astype(np.int32) + 1, self.dim)
        partition = np.sum(np.exp(np.matmul(all_h, np.matmul(self.W, np.transpose(all_h)))))
        return np.sum(np.exp(np.matmul(all_h, np.matmul(self.W, v))), axis=0) / partition


def build_generative_model(g, explore_steps=10000):
    size = g.shape[0]
    dimension = size
    bin_rep = generate_onehot_representation(np.arange(size), dimension)
    model = RBM(dimension)
    for i in range(explore_steps):
        path = random_path(g, random.randint(0, g.shape[0] - 1), random.randint(0, g.shape[0] - 1))
        if len(path) > 0:
            model.incrementally_learn(np.transpose(bin_rep[path]))
    return model


def count_in_betweeness_centrality(g, explore_steps=2000):
    stat = np.zeros([g.shape[0]])
    for i in range(explore_steps):
        path = random_path(g, random.randint(0, g.shape[0] - 1), random.randint(0, g.shape[0] - 1))
        stat[path] = stat[path] + 1

    return stat / np.sum(stat)


def build_energy_model(g, explore_steps=10000):
    size = g.shape[0]
    dimension = size
    bin_rep = generate_onehot_representation(np.arange(size), dimension)
    model = energy.Energy_model(dimension)
    for i in range(explore_steps):
        path = random_path(g, random.randint(0, g.shape[0] - 1), random.randint(0, g.shape[0] - 1))
        if len(path) > 1:
            model.incrementally_learn(np.transpose(bin_rep[path[1:]]), np.transpose(bin_rep[path[:-1]]))
    return model


def normalize(seq):
    m = np.mean(seq)
    s = np.std(seq)
    return (seq - m) / s


if __name__ == '__main__':
    # bin_rep = generate_binary_representation(np.arange(size) + 1, 8)
    # print(bin_rep)

    g = random_graph(32, 0.3)
    x = np.arange(g.shape[0])

    # directly compute
    print("compute ground truth")
    data = count_in_betweeness_centrality(g)
    plt.plot(x, normalize(data), 'b')

    # using node degree
    print("compute node degree")
    out_going, in_coming = node_degrees(g)
    degrees = np.minimum(out_going, in_coming)
    data = degrees / np.sum(degrees)
    plt.plot(x, normalize(data), 'b--')

    # # using generative model
    # print("compute generative model")
    # M = build_generative_model(g)
    # bin_rep = generate_onehot_representation(np.arange(g.shape[0]), g.shape[0])
    # data = M.compute_prop(np.transpose(bin_rep))
    # plt.plot(x, normalize(data), 'b:')

    # using energy model and entropy
    print("compute energy model")
    M = build_energy_model(g)
    bin_rep = generate_onehot_representation(np.arange(g.shape[0]), g.shape[0])
    data = M.compute_entropy(np.transpose(bin_rep))
    plt.plot(x, normalize(data), 'g')
    print(g)
    print(M)

    plt.show()
