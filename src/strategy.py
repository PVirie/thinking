import sys
import os
import random
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "networks"))
from networks import energy


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


def build_energy_layer(graph, explore_steps=2000):

    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])

    model_neighbor = energy.Energy_model(graph.shape[0])
    model_estimate = energy.Energy_model(graph.shape[0])

    for i in range(explore_steps):
        path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
        path = all_reps[path]
        entropy = model_neighbor.compute_entropy(path)
        model_neighbor.incrementally_learn(np.transpose(path[:-1, :]), np.transpose(path[1:, :]))
        last_pv = 0
        for j in range(1, path.shape[0]):
            if entropy[j] < entropy[j - 1]:
                last_pv = j - 1
            model_estimate.incrementally_learn(np.transpose(path[last_pv:j, :]), np.transpose(path[j:(j + 1)]))
