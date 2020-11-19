import sys
import os
import random
import numpy as np
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "networks"))
from networks import network


def generate_onehot_representation(d, max_digits=8):
    b = np.zeros((d.size, max_digits))
    b[np.arange(d.size), d] = 1
    return b


def random_graph(size, p):
    raw = np.random.rand(size, size)
    graph = (raw < p).astype(np.int32)
    graph[np.arange(size), np.arange(size)] = 0
    return graph


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


def enhancer(c):
    b = np.zeros((c.shape[0], c.shape[1]))
    b[np.argmax(c, axis=0), np.arange(c.shape[1])] = 1
    return b

def build_energy_hierarchy(graph, explore_steps=2000):

    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])

    root = network.build_network(graph.shape[0], 3, enhancer)

    for i in range(explore_steps):
        path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
        path = all_reps[path, :]
        root.incrementally_learn(np.transpose(path))

    return root, all_reps


if __name__ == '__main__':

    g = random_graph(64, 0.1)
    cognitive_map, representations = build_energy_hierarchy(g, 10000)
    print("Finish learning.")

    goals = random.sample(range(g.shape[0]), g.shape[0] // 4)

    total_length = 0
    stamp = time.time()
    for t in goals:
        p = cognitive_map.find_path(np.transpose(representations[0:1, :]), np.transpose(representations[t:(t + 1), :]))
        p = list(p)
        total_length = total_length + len(p)
        print([np.argmax(n)[0] for n in p])
    print("energy planner:", time.time() - stamp, " average length:", total_length / len(goals))
