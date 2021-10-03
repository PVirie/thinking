import sys
import os
import random
import numpy as np
import torch


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))
sys.path.append(os.path.join(dir_path, "..", "models"))
sys.path.append(os.path.join(dir_path, "..", "trainers"))

from utilities import *
from models import spline_flow as embedding
from trainers import mse_loss_trainer as trainer


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

    trainer = trainer.Trainer(
        embedding_model=model,
        lr=0.0001, step_size=1000, weight_decay=0.99
    )

    for i in range(10000):
        path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
        loss, _ = trainer.incrementally_learn(all_reps[:, path])

        if i % 500 == 0:
            print("Result after", i, "steps", loss)
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
