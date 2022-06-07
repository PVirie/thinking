import random
import numpy as np
import time
import network
from utilities import *


def load_cognitive_map(config, weight_path=None):
    with network.build_network(config, weight_path, save_on_exit=train) as root:
        pass
    return root


def build_cognitive_map(graph, all_reps, config, explore_steps=2000, weight_path=None):
    with network.build_network(config, weight_path, save_on_exit=train) as root:
        print("Training a cognitive map:")
        for i in range(explore_steps):
            path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
            path = all_reps[:, path]
            root.incrementally_learn(path)
            if i % (explore_steps // 100) == 0:
                print("Training progress: %.2f%%" % (i * 100 / explore_steps), end="\r", flush=True)
        print("Finish learning.")

    return root


if __name__ == '__main__':
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(dir_path, "..", "weights", "strategy.py.test")

    graph_shape = 16

    config = {
        "layers": [
            {
                "num_dimensions": graph_shape,
                "memory_slots": 512,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            },
            {
                "num_dimensions": graph_shape,
                "memory_slots": 256,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            },
            {
                "num_dimensions": graph_shape,
                "memory_slots": 128,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            }
        ]
    }

    representations = generate_onehot_representation(np.arange(graph_shape), graph_shape)

    answer = input("Do you want to retrain? (y/n): ").lower().strip()
    train = answer == "y"

    if train:
        empty_directory(weight_path)

        g = random_graph(graph_shape, 0.2)
        np.save(os.path.join(weight_path, "graph.npy"), g)
        print(g)

        cognitive_map = build_cognitive_map(g, representations, config, 2000, weight_path=weight_path)
        print(cognitive_map)
    else:

        g = np.load(os.path.join(weight_path, "graph.npy"))
        print(g)

        cognitive_map = load_cognitive_map(config, weight_path=weight_path)
        print(cognitive_map)

    goals = range(graph_shape)
    max_steps = 40

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps)
            p = list(p)
            total_length = total_length + len(p)
            print([np.argmax(n) for n in p], t)
        except RecursionError:
            total_length = total_length + max_steps
            print("fail to find path in time.", t)
    print("cognitive planner:", time.time() - stamp, " average length:", total_length / len(goals))

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps, pathway_bias=-1)
            p = list(p)
            total_length = total_length + len(p)
        except RecursionError:
            total_length = total_length + max_steps
    print("hippocampus planner:", time.time() - stamp, " average length:", total_length / len(goals))

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps, pathway_bias=1)
            p = list(p)
            total_length = total_length + len(p)
        except RecursionError:
            total_length = total_length + max_steps
    print("cortex planner:", time.time() - stamp, " average length:", total_length / len(goals))

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            p = shortest_path(g, 0, t)
            p = list(reversed(p))
            total_length = total_length + len(p)
            print(p)
        except Exception:
            total_length = total_length + max_steps

    print("optimal planner:", time.time() - stamp, " average length:", total_length / len(goals))
