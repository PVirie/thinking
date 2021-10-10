import sys
import os
import random
import numpy as np
import torch
import mse_loss_trainer as trainer


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))
sys.path.append(os.path.join(dir_path, "..", "embeddings"))
sys.path.append(os.path.join(dir_path, "..", "variationals"))

from utilities import *
from embeddings import spline_flow as embedding
from variationals import gaussian_variational as variational


def mat_sqr_diff(A, B):
    A_ = np.transpose(A)
    return np.mean(np.square(A_), axis=1, keepdims=True) - 2 * np.matmul(A_, B) / A.shape[0] + np.mean(np.square(B), axis=0, keepdims=True)


def sample_path(all_reps, model, s, length):
    path = [s]
    for i in range(length):
        c = torch.from_numpy(all_reps[:, s:s + 1])
        decoded_rep = model.decode(model.encode(c) + torch.normal(torch.zeros_like(c)))
        distances = mat_sqr_diff(all_reps, decoded_rep.detach().cpu().numpy())
        s = np.argmin(distances, axis=1)[0]
        path.append(s)
    return path


if __name__ == '__main__':

    graph = random_graph(16, 0.2)
    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])

    print(graph)

    model = embedding.Model(**{
        'dims': graph.shape[0]
    })

    neighbor = variational.Model(**{
        'dims': graph.shape[0]
    })

    heuristic = variational.Model(**{
        'dims': graph.shape[0]
    })

    trainer = trainer.Trainer(
        embedding_model=model,
        neighbor_variational_model=neighbor,
        heuristic_variational_model=heuristic,
        lr=0.0001, step_size=1000, weight_decay=0.95
    )

    paths = []
    for i in range(1000):
        path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
        paths.append(path)

    for i in range(20):
        for path in paths:
            loss, _ = trainer.incrementally_learn(all_reps[:, path], np.array([len(path) - 1], dtype=np.int64))

        print("Result after", i + 1, "epoches", loss)
        new_metric = model.encode(torch.from_numpy(all_reps))
        reconstruction = model.decode(new_metric).detach().cpu().numpy()
        print("Reconstruction score", np.mean(np.square(all_reps - reconstruction)))
        print("########################################")

    model.eval()

    encoded_rep = model.encode(torch.from_numpy(all_reps))
    decoded_rep = model.decode(encoded_rep)

    distances = mat_sqr_diff(all_reps, decoded_rep.detach().cpu().numpy())
    print("If the next line is sorted, reconstruction is good.")
    print(np.argmin(distances, axis=1))

    distances = mat_sqr_diff(encoded_rep.detach().cpu().numpy(), encoded_rep.detach().cpu().numpy())
    print(distances)

    print(sample_path(all_reps, model, 0, 10))
