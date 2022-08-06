import numpy as np
from contextlib import contextmanager
import hippocampus
import heuristic
import os


def is_same_node(c, t):
    '''
    c start node as shape: [vector length, 1]
    t target node as shape: [vector length, 1]
    '''
    dist = np.linalg.norm(c - t)
    return dist < 1e-4


class Layer:
    def __init__(self, num_dimensions, heuristic_variational_model, memory_slots=2048, chunk_size=16, diminishing_factor=0.9):
        self.num_dimensions = num_dimensions
        self.hippocampus = hippocampus.Hippocampus(self.num_dimensions, memory_slots, chunk_size, diminishing_factor)
        self.heuristic_variational_model = heuristic_variational_model
        self.next = None

    def load(self, weight_path):
        self.heuristic_variational_model.load(os.path.join(weight_path, "heuristic"))
        self.hippocampus.load(os.path.join(weight_path, "hippocampus"))

    def save(self, weight_path):
        self.heuristic_variational_model.save(os.path.join(weight_path, "heuristic"))
        self.hippocampus.save(os.path.join(weight_path, "hippocampus"))

    def __str__(self):
        if self.next is not None:
            return "Num dims: " + str(self.num_dimensions) + "\n" + str(self.next)
        return "Num dims: " + str(self.num_dimensions)

    def assign_next(self, next_layer):
        self.next = next_layer

    def incrementally_learn(self, path):
        '''
        path = [dimensions, batch]
        '''
        if path.shape[1] < 2:
            return

        self.hippocampus.incrementally_learn(path)

        entropy = self.hippocampus.compute_entropy(path)
        all_pvs = []
        for j in range(0, path.shape[1] - 1):
            if j > 0 and entropy[j] < entropy[j - 1]:
                all_pvs.append(j - 1)
        all_pvs.append(path.shape[1] - 1)

        self.heuristic_variational_model.incrementally_learn(path, np.array(all_pvs, dtype=np.int64))

        if self.next is not None and len(all_pvs) > 1:
            self.next.incrementally_learn(path[:, all_pvs])

    def to_next(self, c, forward=True):
        # not just enhance, but select the closest with highest entropy
        # c has shape [dimensions, 1]
        if forward:
            next_candidates = self.hippocampus.get_next()
        else:
            next_candidates = self.hippocampus.get_prev()
        for i in range(1000):
            prev_c = c
            props = self.hippocampus.match(c)
            c = np.sum(np.squeeze(props) * next_candidates / np.sum(props), axis=1, keepdims=True)
            if self.hippocampus.compute_entropy(c) < self.hippocampus.compute_entropy(prev_c):
                return prev_c
        return None

    def from_next(self, c):
        return c

    def pincer_inference(self, s, t, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        candidates, props = self.hippocampus.get_distinct_next_candidate(s)

        cortex_rep, cortex_prop = self.heuristic_variational_model.consolidate(s, candidates, np.squeeze(props), t)
        hippocampus_rep, hippocampus_prop = self.hippocampus.infer(s, t)

        if pathway_bias < 0:
            return hippocampus_rep, []
        if pathway_bias > 0:
            return cortex_rep, []

        compare_results = hippocampus_prop > cortex_prop
        results = np.where(compare_results, hippocampus_rep, cortex_rep)
        return results, [(cortex_rep, cortex_prop), (hippocampus_rep, hippocampus_prop)]

    def find_path(self, c, t, hard_limit=20, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        if self.next is not None:
            goals = self.next.find_path(self.to_next(c, True), self.to_next(t, False))

        count_steps = 0
        yield c, []

        while True:
            if is_same_node(c, t):
                break

            if self.next is not None:
                try:
                    ng, _ = next(goals)
                    g = self.from_next(ng)
                except StopIteration:
                    g = t
            else:
                g = t

            while True:
                if is_same_node(c, t):
                    # wait we found the true target!
                    break

                if is_same_node(c, g):
                    c = g
                    break

                count_steps = count_steps + 1
                if count_steps >= hard_limit:
                    raise RecursionError
                    break

                c, supplimentary = self.pincer_inference(c, g, pathway_bias)
                c = self.hippocampus.enhance(c)  # enhance signal preventing signal degradation

                yield c, supplimentary

    def next_step(self, c, t, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        if is_same_node(c, t):
            return t

        next_g, _ = self.next.next_step(self.to_next(c), self.to_next(t))
        g = self.from_next(next_g)

        if is_same_node(c, g):
            return g

        c, supplimentary = self.pincer_inference(c, g, pathway_bias)
        return c, supplimentary


@contextmanager
def build_network(config, weight_path=None, save_on_exit=True):
    # The following runs BEFORE with block.
    layers = []
    for layer in config["layers"]:
        heuristic_model_params = layer["heuristic_model_param"]
        heuristic_model_params["diminishing_factor"] = layer["diminishing_factor"]
        heuristic_model_params["world_update_prior"] = config["world_update_prior"]
        heuristic_model_params["dims"] = layer["num_dimensions"]
        heuristic_model = heuristic.Model(**heuristic_model_params)
        layers.append(Layer(layer["num_dimensions"], heuristic_model, layer["memory_slots"], layer["chunk_size"], layer["diminishing_factor"]))

    for i in range(len(layers) - 1):
        layers[i].assign_next(layers[i + 1])

    for i, layer in enumerate(layers):
        layer.load(os.path.join(weight_path, str(i)))

    # The following returns into the with block.
    yield layers[0]

    # The following runs AFTER with block.
    if save_on_exit:
        for i, layer in enumerate(layers):
            layer.save(os.path.join(weight_path, str(i)))


if __name__ == '__main__':
    print("assert that probabilistic network works.")
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [0]])))
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [2]])))
    print(is_same_node(np.array([[1], [0]]), np.array([[1], [0]])))
