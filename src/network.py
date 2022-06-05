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
    def __init__(self, num_dimensions, heuristic_variational_model, memory_slots=2048):
        self.num_dimensions = num_dimensions
        self.hippocampus = hippocampus.Hippocampus(self.num_dimensions, memory_slots)
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

    def to_next(self, c):
        # should not just enhance, but select the closest with highest entropy
        return self.hippocampus.enhance(c)

    def from_next(self, c):
        return c

    def pincer_inference(self, s, t):
        props = self.hippocampus.match(s)
        candidates = self.hippocampus.get_next()

        cortex_rep, cortex_prop = self.heuristic_variational_model.consolidate(candidates, np.squeeze(props), t)
        hippocampus_rep, hippocampus_prop = self.hippocampus.infer(s, t)

        compare_results = hippocampus_prop > cortex_prop
        results = np.where(compare_results, hippocampus_rep, cortex_rep)
        return results

    def find_path(self, c, t, hard_limit=20):

        if self.next is not None:
            goals = self.next.find_path(self.to_next(c), self.to_next(t))

        count_steps = 0
        yield c
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

                c = self.pincer_inference(c, g)
                c = self.hippocampus.enhance(c)  # enhance signal preventing signal degradation

                yield c


@contextmanager
def build_network(config, weight_path=None, save_on_exit=True):
    # The following runs BEFORE with block.
    layers = []
    for layer in config["layers"]:
        heuristic_model_params = layer["heuristic_model_param"]
        heuristic_model = heuristic.Model(**heuristic_model_params)
        layers.append(Layer(layer["num_dimensions"], heuristic_model, layer["memory_slots"]))

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
