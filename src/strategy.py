import random
import numpy as np
import time
import network
from utilities import *


def load_cognitive_map(config, weight_path=None, logger=None):
    with network.build_network(config, weight_path, save_on_exit=train, logger=logger) as root:
        pass
    return root


def build_cognitive_map(graph, all_reps, config, explore_steps=2000, weight_path=None, logger=None):
    with network.build_network(config, weight_path, save_on_exit=train, logger=logger) as root:
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
    np.set_printoptions(precision=2)

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(dir_path, "..", "weights", "strategy.py.test")
    os.makedirs(weight_path, exist_ok=True)

    import json
    artifact_path = os.path.join(dir_path, "..", "artifacts")
    os.makedirs(artifact_path, exist_ok=True)

    log_data = []
    path_details = None

    def logger(data):
        global path_details

        if path_details is None:
            path_details = []

        if "success" in data:
            if data["layer"] == 0:
                log_data.append(path_details)
                path_details = None
            return
        elif "s" in data:
            path_details.append({
                "layer": data["layer"],
                "s": int(np.argmax(data["s"], axis=0)[0]),
                "t": int(np.argmax(data["t"], axis=0)[0]),
            })
        else:
            layer = data["layer"]
            selected = int(np.argmax(data["selected"], axis=0)[0])
            choices = [(int(np.argmax(rep, axis=0)[0]), float(prop[0])) for rep, prop in data["choices"]]
            path_details.append({
                "layer": layer,
                "selected": selected,
                "choices": choices
            })

    graph_shape = 16

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

    representations = generate_onehot_representation(np.arange(graph_shape), graph_shape)

    answer = input("Do you want to retrain? (y/n): ").lower().strip()
    train = answer == "y"

    if train:
        empty_directory(weight_path)

        g = random_graph(graph_shape, 0.2)
        np.save(os.path.join(weight_path, "graph.npy"), g)
        print(g)

        cognitive_map = build_cognitive_map(g, representations, config, 2000, weight_path=weight_path, logger=logger)
        print(cognitive_map)
    else:

        g = np.load(os.path.join(weight_path, "graph.npy"))
        print(g)

        cognitive_map = load_cognitive_map(config, weight_path=weight_path, logger=logger)
        print(cognitive_map)

    goals = range(graph_shape)
    max_steps = 40

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps)
            for pi in p:
                print(np.argmax(pi), end=' ')
                total_length = total_length + 1
        except RecursionError:
            print("fail to find path in time.", t, end=' ')
        finally:
            print()
    print("cognitive planner:", time.time() - stamp, " average length:", total_length / len(goals))

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps, pathway_bias=-1)
            for pi in p:
                print(np.argmax(pi), end=' ')
                total_length = total_length + 1
        except RecursionError:
            print("fail to find path in time.", t, end=' ')
        finally:
            print()
    print("hippocampus planner:", time.time() - stamp, " average length:", total_length / len(goals))

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps, pathway_bias=1)
            for pi in p:
                print(np.argmax(pi), end=' ')
                total_length = total_length + 1
        except RecursionError:
            print("fail to find path in time.", t, end=' ')
        finally:
            print()
    print("cortex planner:", time.time() - stamp, " average length:", total_length / len(goals))

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            p = shortest_path(g, 0, t)
            p = list(reversed(p))
            for pi in p:
                print(pi, end=' ')
                total_length = total_length + 1
        except RecursionError:
            print("fail to find path in time.", t, end=' ')
        finally:
            print()

    print("optimal planner:", time.time() - stamp, " average length:", total_length / len(goals))

    with open(os.path.join(artifact_path, "strategy_results.json"), "w") as file:
        json.dump(log_data, file)
