import sys
import os
import random
import numpy as np

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


if __name__ == '__main__':

    import torch

    graph = random_graph(16, 0.2)
    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])

    print(graph)

    model = embedding.Model(**{
        'dims': graph.shape[0]
    })

    neighbor_model = embedding.Model(**{
        'dims': graph.shape[0]
    })

    trainer = trainer.Trainer(
        embedding_model=model,
        neighbor_model=neighbor_model,
        lr=0.001, step_size=1000, weight_decay=0.99
    )

    for i in range(5000):
        path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
        loss, _ = trainer.incrementally_learn(all_reps[:, path])

        if i % 500 == 0:
            print("Result after", i, "steps", loss)
            new_metric = model.encode(torch.from_numpy(all_reps))
            reconstruction = model.decode(new_metric).detach().cpu().numpy()
            print("Reconstruction score", np.mean(np.square(all_reps - reconstruction)))
            print("########################################")

    model.eval()
    neighbor_model.eval()

    encoded_rep = model.encode(torch.from_numpy(all_reps))
    encoded_next_rep = neighbor_model.encode(encoded_rep)
    next_rep = model.decode(encoded_next_rep)

    print(encoded_rep.shape, encoded_next_rep.shape)

    distances = mat_sqr_diff(encoded_rep.detach().cpu().numpy(), encoded_next_rep.detach().cpu().numpy())
    print(distances)
