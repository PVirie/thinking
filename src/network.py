import numpy as np
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "models"))
sys.path.append(os.path.join(dir_path, "hierarchy"))

from models import energy


def is_same_node(c, t):
    '''
    c start node as shape: [vector length, 1]
    t target node as shape: [vector length, 1]
    '''
    return np.linalg.norm(c - t) < 1e-2


class Layer:
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions
        self.model_neighbor = energy.Energy_model(self.num_dimensions, negative_init=False)
        self.model_estimate = energy.Energy_model(self.num_dimensions, negative_init=False)
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
        self.model_neighbor.incrementally_learn(path[:, :-1], path[:, 1:])

        entropy = self.model_neighbor.compute_entropy(path)
        last_pv = 0
        all_pvs = []
        for j in range(0, path.shape[1]):
            if j > 0 and entropy[j] < entropy[j - 1]:
                last_pv = j - 1  # remove this line will improve the result, but cost the network more.
                all_pvs.append(j - 1)
            self.model_estimate.incrementally_learn(path[:, last_pv:(j + 1)], path[:, j:(j + 1)])

        if self.next is not None and len(all_pvs) > 1:
            self.next.incrementally_learn(path[:, all_pvs])

    def encode(self, c, forward=True):
        c_ent = self.model_neighbor.compute_entropy(c)
        while True:
            if forward:
                n = self.model_neighbor.forward(c)
            else:
                n = self.model_neighbor.backward(c)
            n_ent = self.model_neighbor.compute_entropy(n)
            if c_ent > n_ent:
                return c
            else:
                c = n
                c_ent = n_ent

    def decode(self, c):
        return c

    def find_path(self, c, t, hard_limit=20):
        if self.next is not None:
            goals = self.next.find_path(self.next.encode(c, forward=True), self.next.encode(t, forward=False))

        count_steps = 0
        yield c
        while True:
            if is_same_node(c, t):
                break

            if self.next is not None:
                try:
                    g = self.next.decode(next(goals))
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
                c = self.model_neighbor.pincer_inference(self.model_neighbor, self.model_estimate, c, g)
                yield c

            c = g


def build_network(num_dimensions, num_layers):

    root = Layer(num_dimensions)
    last = root
    for i in range(num_layers - 1):
        temp = Layer(num_dimensions)
        last.assign_next(temp)
        last = temp

    return root


if __name__ == '__main__':
    print("assert that probabilistic network works.")
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [0]])))
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [2]])))
