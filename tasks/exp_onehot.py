import random
import numpy as np
import time
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "..", "src"))

import network
from utilities import *


def build_cognitive_map(graph, all_reps, config, explore_steps=2000, weight_path=None, logger=None):
    with network.build_network(config, weight_path, save_on_exit=False, load_exists=False) as root:
        for i in range(explore_steps):
            path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
            path = all_reps[:, path]
            root.incrementally_learn(path)
    return root


def do_experiment(config, result_file, repeat):

    print("Start experiment.")

    results = {}

    def log_result(planner, elapsed, ave_length):
        if planner not in results:
            results[planner] = {
                "elapsed": 0,
                "average length": 0
            }
        results[planner]["elapsed"] += elapsed
        results[planner]["average length"] += ave_length

    for i in range(repeat):
        print("Progress: %.2f%%" % (i * 100 / repeat), end="\r", flush=True)

        g = random_graph(graph_shape, 0.2)
        cognitive_map = build_cognitive_map(g, representations, config, 2000, weight_path=weight_path)

        goals = range(graph_shape)
        max_steps = 40

        total_length = 0
        stamp = time.time()
        for t in goals:
            try:
                p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps)
                for pi in p:
                    total_length = total_length + 1
            except RecursionError:
                pass
        log_result("cognitive planner", time.time() - stamp, total_length / len(goals))

        total_length = 0
        stamp = time.time()
        for t in goals:
            try:
                p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps, pathway_bias=-1)
                for pi in p:
                    total_length = total_length + 1
            except RecursionError:
                pass
        log_result("hippocampus planner", time.time() - stamp, total_length / len(goals))

        total_length = 0
        stamp = time.time()
        for t in goals:
            try:
                p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps, pathway_bias=1)
                for pi in p:
                    total_length = total_length + 1
            except RecursionError:
                pass
        log_result("cortex planner", time.time() - stamp, total_length / len(goals))

        total_length = 0
        stamp = time.time()
        for t in goals:
            try:
                p = shortest_path(g, 0, t)
                p = list(reversed(p))
                for pi in p:
                    total_length = total_length + 1
            except RecursionError:
                pass
        log_result("optimal planner", time.time() - stamp, total_length / len(goals))

    config["results"] = results

    with open(result_file, "w") as file:
        json.dump(config, file)

    print("Finish experiment.")


if __name__ == '__main__':
    np.set_printoptions(precision=2)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(dir_path, "..", "weights", "exp_onehot.py.test")
    os.makedirs(weight_path, exist_ok=True)

    import json
    artifact_path = os.path.join(dir_path, "..", "artifacts")
    os.makedirs(artifact_path, exist_ok=True)

    graph_shape = 16
    representations = generate_onehot_representation(np.arange(graph_shape), graph_shape)

    config = {
        "world_update_prior": 0.1,
        "layers": [
            {
                "num_dimensions": graph_shape,
                "memory_slots": 64,
                "chunk_size": 16,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'pre_steps': 4, 'all_pairs': False,
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            },
            {
                "num_dimensions": graph_shape,
                "memory_slots": 64,
                "chunk_size": 12,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'pre_steps': 4, 'all_pairs': False,
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            },
            {
                "num_dimensions": graph_shape,
                "memory_slots": 64,
                "chunk_size": 8,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'pre_steps': 4, 'all_pairs': False,
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            }
        ]
    }

    do_experiment(config, os.path.join(artifact_path, "exp_onehot_results.json"), repeat=2)
