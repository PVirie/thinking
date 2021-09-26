import sys
import os
import random
import numpy as np
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))
sys.path.append(os.path.join(dir_path, "..", "models"))
sys.path.append(os.path.join(dir_path, "..", "trainers"))

from models import spline_flow as embedding
from trainers import mse_loss_trainer as trainer


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


def measure_distance(graph, representations):
    # print(representations)
    grid_x, grid_y = np.meshgrid(np.arange(graph.shape[0]), np.arange(graph.shape[1]))
    X = grid_x.flatten()[graph.flatten()]
    Y = grid_y.flatten()[graph.flatten()]
    S = representations[:, Y]
    T = representations[:, X]
    print("Neighbor score:", np.mean(np.square(S - T)))

    X = grid_x.flatten()
    Y = grid_y.flatten()
    S = representations[:, Y]
    T = representations[:, X]
    print("Average score:", np.mean(np.square(S - T)))


if __name__ == '__main__':

    import torch

    graph = random_graph(16, 0.2)
    print("Result using the adjacency matrix itself as the representation.")
    measure_distance(graph, np.transpose(graph))
    print("########################################")

    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])
    print("Result using one-hot representation.")
    measure_distance(graph, all_reps)
    print("########################################")

    model = embedding.Model(**{
        'dims': graph.shape[0]
    })

    neighbor_model = embedding.Model(**{
        'dims': graph.shape[0]
    })

    trainer = trainer.Trainer(
        embedding_model=model,
        neighbor_model=neighbor_model,
        lr=0.01, step_size=1000, weight_decay=0.99
    )

    for i in range(2000):
        path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
        trainer.incrementally_learn(all_reps[:, path])

        if i % 100 == 0:
            print("Result after", i, "steps")
            new_metric = model.encode(torch.from_numpy(all_reps))
            measure_distance(graph, new_metric.detach().cpu().numpy())

            reconstruction = model.decode(new_metric).detach().cpu().numpy()
            print("Reconstruction score", np.mean(np.square(all_reps - reconstruction)))
            print("########################################")
