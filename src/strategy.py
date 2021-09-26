import random
import numpy as np
import time
import network


def generate_onehot_representation(d, max_digits=8):
    b = np.zeros((max_digits, d.size), dtype=np.float32)
    b[d, np.arange(d.size)] = 1
    return b


def random_graph(size, p):
    raw = np.random.rand(size, size)
    graph = (raw < p).astype(np.int32)
    graph[np.arange(size), np.arange(size)] = 0
    return graph


def random_walk(graph, s, max_steps):
    unvisited = np.ones([graph.shape[0]], dtype=bool)
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


def build_energy_hierarchy(graph, explore_steps=2000, weight_path=None):

    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])

    config = {
        "layers": [
            {
                "num_dimensions": graph.shape[0],
                "memory_slots": 1024,
                "embedding_model": "spline_flow",
                "neighbor_model": "spline_flow",
                "trainer": {
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            }
        ]
    }

    with network.build_network(config, save_on_exit=False) as root:
        for i in range(explore_steps):
            path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
            path = all_reps[:, path]
            root.incrementally_learn(path)

    return root, all_reps


if __name__ == '__main__':

    g = random_graph(16, 0.2)
    print(g)

    cognitive_map, representations = build_energy_hierarchy(g, 2000)
    print("Finish learning.")
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
