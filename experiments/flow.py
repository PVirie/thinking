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


def build_energy_model(g, explore_steps=2000):
    size = g.shape[0]
    dimension = math.ceil(math.log(size, 2))
    bin_rep = generate_binary_representation(np.arange(size) + 1, dimension)
    M = np.zeros([dimension, dimension])
    for i in range(explore_steps):
        path = list(reversed(random_path(g, 0, size - 1)))
        temp_bin_rep = bin_rep[path]
        for j in range(len(path) - 1):
            m = np.matmul(np.reshape(temp_bin_rep[j], [-1, 1]), np.reshape(temp_bin_rep[j+1], [1, -1]))
            n = np.matmul(np.reshape(bin_rep[random.randint(0, size-1)], [-1, 1]), np.reshape(bin_rep[random.randint(0, size-1)], [1, -1]))
            M = M + 0.001*(m - n)
    return M


if __name__ == '__main__':
    # bin_rep = generate_binary_representation(np.arange(size) + 1, 8)
    # print(bin_rep)

    g = random_graph(128, 0.1)
    M = build_energy_model(g)
    print(M)