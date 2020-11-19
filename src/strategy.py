import sys
import os
import random
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "networks"))
from networks import network


def generate_onehot_representation(d, max_digits=8):
    b = np.zeros((d.size, max_digits))
    b[np.arange(d.size), d] = 1
    return b


def random_walk(graph, s, max_steps):
    unvisited = np.ones([graph.shape[0]], dtype=np.bool)
    indices = np.arange(graph.shape[0])
    path = []
    c = s
    for i in range(max_steps):
        path.append(c)
        unvisited[c] = False
        candidates = indices[np.logical_and((graph[c, :] > 0), unvisited)]
        if candidates.shape[0] == 0:
            return path
        candidate_weights = np.exp(-graph[c, candidates])  # graph contains costs
        candidate_weights = candidate_weights / np.sum(candidate_weights)
        c = np.random.choice(candidates, 1, p=candidate_weights)[0]
    return path


def build_energy_hierarchy(graph, explore_steps=2000):

    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])

    root = network.build_network(graph.shape[0], 3)

    for i in range(explore_steps):
        path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
        path = all_reps[path]
        root.incrementally_learn(path)

    return root
