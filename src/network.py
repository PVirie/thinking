import numpy as np
import sys
import os
import importlib
from contextlib import contextmanager
import torch

import variational_energy
import hippocampus

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "models"))
from models import embedding_base
from trainers.mse_loss_trainer import Trainer


def is_same_node(c, t):
    '''
    c start node as shape: [vector length, 1]
    t target node as shape: [vector length, 1]
    '''
    return np.linalg.norm(c - t) < 1e-4


class Layer:
    def __init__(self, num_dimensions, memory_slots=2048, embedding_model=None, neighbor_model=None, trainer=None):
        self.num_dimensions = num_dimensions
        self.neighbor_distribution = variational_energy.Energy_model(self.num_dimensions)
        self.estimate_distribution = variational_energy.Energy_model(self.num_dimensions)
        self.hippocampus = hippocampus.Hippocampus(self.num_dimensions, memory_slots)
        self.embedding = embedding_model if embedding_model is not None else embedding_base.Model(num_dimensions)
        self.neighbor_model = neighbor_model if neighbor_model is not None else embedding_base.Model(num_dimensions)
        self.trainer = trainer if trainer is not None else Trainer(embedding_model, neighbor_model)
        self.next = None

    def __str__(self):
        if self.next is not None:
            return str(self.neighbor_distribution) + "\n" + str(self.next)
        return str(self.neighbor_distribution)

    def assign_next(self, next_layer):
        self.next = next_layer

    def incrementally_learn(self, path):
        '''
        path = [dimensions, batch]
        '''
        if path.shape[1] < 2:
            return

        # Learn embedding and forward model
        self.trainer.incrementally_learn(path)

        # learning distribution
        self.embedding.eval()
        self.neighbor_model.eval()
        encoded_path = self.embedding.encode(path)

        self.neighbor_distribution.incrementally_learn(self.forward(encoded_path[:, :-1]), encoded_path[:, 1:])
        self.hippocampus.incrementally_learn(encoded_path)

        entropy = self.hippocampus.compute_entropy(encoded_path)
        last_pv = 0
        all_pvs = []
        for j in range(0, encoded_path.shape[1]):
            if j > 0 and entropy[j] < entropy[j - 1]:
                last_pv = j - 1  # remove this line will improve the result, but cost the network more.
                all_pvs.append(j - 1)
            self.estimate_distribution.incrementally_learn(encoded_path[:, last_pv:(j + 1)], encoded_path[:, j:(j + 1)])

        if self.next is not None and len(all_pvs) > 1:
            self.next.incrementally_learn(encoded_path[:, all_pvs])

    def forward(self, c):
        return self.neighbor_model.encode(c)

    def backward(self, t):
        return self.neighbor_model.decode(t)

    def to_next(self, c, forward=True):
        c_ent = self.hippocampus.compute_entropy(c)
        while True:
            if forward:
                n = self.forward(c)
            else:
                n = self.backward(c)
            n_ent = self.hippocampus.compute_entropy(n)
            if c_ent > n_ent:
                return c
            else:
                c = n
                c_ent = n_ent

    def from_next(self, c):
        return c

    def find_path(self, c, t, hard_limit=20):
        c = self.embedding.encode(c)
        t = self.embedding.encode(t)

        if self.next is not None:
            goals = self.next.find_path(self.to_next(c, forward=True), self.to_next(t, forward=False))

        count_steps = 0
        yield self.embedding.decode(c)
        while True:
            if is_same_node(c, t):
                break

            if self.next is not None:
                try:
                    g = self.from_next(next(goals))
                except StopIteration:
                    g = t
            else:
                g = t

            while True:
                count_steps = count_steps + 1
                if count_steps >= hard_limit:
                    raise RecursionError
                if is_same_node(c, g):
                    break
                x = self.forward(c)
                c, _ = self.hippocampus.pincer_inference(self.neighbor_distribution, self.estimate_distribution, c, x, g)
                yield self.embedding.decode(c)

            c = g


@contextmanager
def build_network(config, save_on_exit=True):
    # The following runs BEFORE with block.
    layers = []
    for layer in config["layers"]:
        embedding_model = importlib.import_module(layer["embedding_model"], package="models").Model(layer["num_dimensions"])
        neighbor_model = importlib.import_module(layer["neighbor_model"], package="models").Model(layer["num_dimensions"])
        trainer_params = layer["trainer"]
        trainer_params["embedding_model"] = embedding_model
        trainer_params["neighbor_model"] = neighbor_model
        trainer = Trainer(**trainer_params)
        layers.append(Layer(layer["num_dimensions"], layer["memory_slots"], embedding_model, neighbor_model, trainer))

    for i in range(len(layers) - 1):
        layers[i].assign_next(layers[i + 1])

    for layer in layers:
        layer.trainer.load()

    # The following returns into the with block.
    yield layers[0]

    # The following runs AFTER with block.
    if save_on_exit:
        for layer in layers:
            layer.trainer.save()


if __name__ == '__main__':
    print("assert that probabilistic network works.")
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [0]])))
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [2]])))
    print(is_same_node(np.array([[1], [0]]), np.array([[1], [0]])))
