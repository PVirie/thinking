import random
import numpy as np
import time
import network
from utilities import *


def build_energy_hierarchy(graph, explore_steps=2000, train=None, weight_path=None):

    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])

    config = {
        "layers": [
            {
                "num_dimensions": graph.shape[0],
                "memory_slots": 1024,
                "heuristic_model_param": {
                    'dims': graph.shape[0], 'lr': 0.0001, 'step_size': 1000, 'weight_decay': 0.99
                }
            },
            {
                "num_dimensions": graph.shape[0],
                "memory_slots": 256,
                "heuristic_model_param": {
                    'dims': graph.shape[0], 'lr': 0.0001, 'step_size': 1000, 'weight_decay': 0.99
                }
            }
        ]
    }

    if train is None:
        answer = input("Do you want to retrain? (y/n): ").lower().strip()
        train = answer == "y"

    if train:
        empty_directory(weight_path)

    with network.build_network(config, weight_path, save_on_exit=train) as root:
        if train:
            print("Training a cognitive map:")
            for i in range(explore_steps):
                path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
                path = all_reps[:, path]
                root.incrementally_learn(path)
                if i % (explore_steps // 100) == 0:
                    print("Training progress: %.2f%%" % (i * 100 / explore_steps), end="\r", flush=True)
            print("Finish learning.")

    return root, all_reps


if __name__ == '__main__':
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    g = random_graph(16, 0.2)
    print(g)

    cognitive_map, representations = build_energy_hierarchy(g, 10000, weight_path=os.path.join(dir_path, "..", "weights", "strategy.py.test"))
    print(cognitive_map)

    goals = random.sample(range(g.shape[0]), g.shape[0] // 2)

    total_length = 0
    max_steps = 20
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
    print("energy planner:", time.time() - stamp, " average length:", total_length / len(goals))
