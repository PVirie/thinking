import numpy as np
import sys
import os
import importlib
from contextlib import contextmanager

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "models"))
sys.path.append(os.path.join(dir_path, "embedding"))

from embedding import embedding_base
from models import energy
from models import hippocampus


def is_same_node(c, t):
    '''
    c start node as shape: [vector length, 1]
    t target node as shape: [vector length, 1]
    '''
    return np.linalg.norm(c - t) < 1e-3


class Layer:
    def __init__(self, num_dimensions, embedding=embedding_base.Embedding(), memory_slots=2048):
        self.num_dimensions = num_dimensions
        self.model_neighbor = energy.Energy_model(self.num_dimensions)
        self.model_estimate = energy.Energy_model(self.num_dimensions)
        self.hippocampus = hippocampus.Hippocampus(self.num_dimensions, memory_slots)
        self.embedding = embedding
        self.next = None

    def __str__(self):
        if self.next is not None:
            return str(self.model_neighbor) + "\n" + str(self.next)
        return str(self.model_neighbor)

    def assign_next(self, next_layer):
        self.next = next_layer

    def incrementally_learn(self, path):
        '''
        path = [dimensions, batch]
        '''
        if path.shape[1] < 2:
            return

        # Learn embedding
        loss, iteration = self.embedding.incrementally_learn(path)

        # Learn neighbor and estimator
        path = self.embedding.encode(path)

        self.model_neighbor.incrementally_learn(path[:, :-1], path[:, 1:])

        self.hippocampus.incrementally_learn(path)

        entropy = self.hippocampus.compute_entropy(path)
        last_pv = 0
        all_pvs = []
        for j in range(0, path.shape[1]):
            if j > 0 and entropy[j] < entropy[j - 1]:
                last_pv = j - 1  # remove this line will improve the result, but cost the network more.
                all_pvs.append(j - 1)
            self.model_estimate.incrementally_learn(path[:, last_pv:(j + 1)], path[:, j:(j + 1)])

        if self.next is not None and len(all_pvs) > 1:
            self.next.incrementally_learn(path[:, all_pvs])

    def to_next(self, c, forward=True):
        c_ent = self.hippocampus.compute_entropy(c)
        while True:
            if forward:
                n = self.model_neighbor.forward(c)
            else:
                n = self.model_neighbor.backward(c)
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
                c, _ = self.hippocampus.pincer_inference(self.model_neighbor, self.model_estimate, c, g)
                yield self.embedding.decode(c)

            c = g


@contextmanager
def build_network(config, save_on_exit=True):
    # The following runs BEFORE with block.
    layers = []
    for layer in config["layers"]:
        embedding_module = importlib.import_module(layer["embedding"], package="embedding")
        layers.append(Layer(layer["num_dimensions"], embedding_module.Embedding(**layer["embedding_config"]), layer["memory_slots"]))

    for i in range(len(layers) - 1):
        layers[i].assign_next(layers[i + 1])

    for layer in layers:
        layer.embedding.load()

    # The following returns into the with block.
    yield layers[0]

    # The following runs AFTER with block.
    if save_on_exit:
        for layer in layers:
            layer.embedding.save()


if __name__ == '__main__':
    print("assert that probabilistic network works.")
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [0]])))
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [2]])))
    print(is_same_node(np.array([[1], [0]]), np.array([[1], [0]])))
