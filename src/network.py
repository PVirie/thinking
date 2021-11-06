import numpy as np
import sys
import os
import importlib
from contextlib import contextmanager
import torch
import hippocampus
from embeddings import embedding_base
from variationals import variational_base
from trainers.mse_loss_trainer import Trainer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "embeddings"))
sys.path.append(os.path.join(dir_path, "variationals"))


def is_same_node(c, t):
    '''
    c start node as shape: [vector length, 1]
    t target node as shape: [vector length, 1]
    '''
    return np.linalg.norm(c - t) < 1e-4


class Layer:
    def __init__(self, num_dimensions, memory_slots=2048, embedding_model=None, neighbor_variational_model=None, heuristic_variational_model=None, trainer=None):
        self.num_dimensions = num_dimensions
        self.hippocampus = hippocampus.Hippocampus(self.num_dimensions, memory_slots)
        self.embedding = embedding_model if embedding_model is not None else embedding_base.Model(num_dimensions)
        self.neighbor_variational_model = neighbor_variational_model if neighbor_variational_model is not None else variational_base.Model(num_dimensions)
        self.heuristic_variational_model = heuristic_variational_model if heuristic_variational_model is not None else variational_base.Model(num_dimensions)
        self.trainer = trainer if trainer is not None else Trainer(embedding_model, neighbor_variational_model, heuristic_variational_model)
        self.next = None

    def __str__(self):
        if self.next is not None:
            return str(self.num_dimensions) + "\n" + str(self.next)
        return str(self.num_dimensions)

    def assign_next(self, next_layer):
        self.next = next_layer

    def incrementally_learn(self, path):
        '''
        path = [dimensions, batch]
        '''
        if path.shape[1] < 2:
            return

        self.embedding.eval()
        encoded_path = self.embedding.encode(path)

        self.hippocampus.incrementally_learn(encoded_path)

        entropy = self.neighbor_variational_model.compute_entropy(encoded_path)
        all_pvs = []
        for j in range(0, encoded_path.shape[1]):
            if j > 0 and entropy[j] < entropy[j - 1]:
                all_pvs.append(j - 1)

        self.trainer.incrementally_learn(path, np.array(all_pvs, dtype=np.int64))

        if self.next is not None and len(all_pvs) > 1:
            self.next.incrementally_learn(encoded_path[:, all_pvs])

    def to_next(self, c):
        # should not just enhance, but select the closest with highest entropy
        return self.hippocampus.enhance(c)

    def from_next(self, c):
        return c

    def find_path(self, c, t, hard_limit=20):
        c = self.embedding.encode(c)
        t = self.embedding.encode(t)

        if self.next is not None:
            goals = self.next.find_path(self.to_next(c), self.to_next(t))

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
                    # raise RecursionError
                    break
                if is_same_node(c, g):
                    break
                c, _ = self.neighbor_variational_model.pincer_inference(self.neighbor_variational_model, self.heuristic_variational_model, c, g)
                yield self.embedding.decode(c)

            c = g


@contextmanager
def build_network(config, save_on_exit=True):
    # The following runs BEFORE with block.
    layers = []
    for layer in config["layers"]:
        embedding_model = importlib.import_module(layer["embedding_model"], package="embeddings").Model(layer["num_dimensions"])
        neighbor_variational_model = importlib.import_module(layer["neighbor_variational_model"], package="variationals").Model(layer["num_dimensions"])
        heuristic_variational_model = importlib.import_module(layer["heuristic_variational_model"], package="variationals").Model(layer["num_dimensions"])
        trainer_params = layer["trainer"]
        trainer_params["embedding_model"] = embedding_model
        trainer_params["neighbor_variational_model"] = neighbor_variational_model
        trainer_params["heuristic_variational_model"] = heuristic_variational_model

        trainer = Trainer(**trainer_params)
        layers.append(Layer(layer["num_dimensions"], layer["memory_slots"], embedding_model, neighbor_variational_model, heuristic_variational_model, trainer))

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
