import numpy as np
import energy
import path_finder


def is_same_node(c, t):
    '''
    c start node as shape: [vector length, 1]
    t target node as shape: [vector length, 1]
    '''
    return np.linalg.norm(c - t) < 1e-2


class Layer:
    def __init__(self, num_dimensions, enhancer):
        self.num_dimensions = num_dimensions
        self.model_neighbor = energy.Energy_model(self.num_dimensions)
        self.model_estimate = energy.Energy_model(self.num_dimensions)
        self.enhancer = enhancer
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
        entropy = self.model_neighbor.compute_entropy(path)
        self.model_neighbor.incrementally_learn(path[:, :-1], path[:, 1:], lr=0.01)
        path_log_props = np.cumsum(np.log(self.model_neighbor.compute_prop(path[:, :-1], path[:, 1:]), axis=0))
        last_pv = 0
        last_log_prop = 1.0
        all_pvs = []
        for j in range(1, path.shape[1]):
            if entropy[j] < entropy[j - 1]:
                last_pv = j - 1
                last_log_prop = 1.0 if last_pv == 0 else path_log_props[last_pv - 1]
                all_pvs.append(j - 1)
            self.model_estimate.learn(path[:, last_pv:j], path[:, j:(j + 1)], path_log_props[last_pv:j] - last_log_prop, lr=0.01)

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

    def find_path(self, c, t):
        if self.next is not None:
            goals = self.next.find_path(self.next.encode(c, forward=True), self.next.encode(t, forward=False))

        yield c
        while True:
            if is_same_node(c, t):
                break

            if self.next is not None:
                try:
                    g = self.enhancer(self.next.decode(next(goals)))
                except StopIteration:
                    g = t
            else:
                g = t
            print("goal!", np.argmax(g))
            while True:
                if is_same_node(c, g):
                    break
                c = self.enhancer(energy.pincer_inference(self.model_neighbor, self.model_estimate, c, g))
                yield c

            c = g

    def find_path_breadth(self, c, t):
        if self.next is not None:
            goals = self.next.find_path_breadth(self.next.encode(c, forward=True), self.next.encode(t, forward=False))

        yield c
        while True:
            if is_same_node(c, t):
                break

            if self.next is not None:
                try:
                    g = self.enhancer(self.next.decode(next(goals)))
                except StopIteration:
                    g = t
            else:
                g = t

            p = path_finder.shortest_path(self, is_same_node, c, g)
            for p_i in p:
                yield p_i
            c = g


def build_network(num_dimensions, num_layers, enhancer):

    root = Layer(num_dimensions, enhancer)
    last = root
    for i in range(num_layers - 1):
        temp = Layer(num_dimensions, enhancer)
        last.assign_next(temp)
        last = temp

    return root


if __name__ == '__main__':
    print("assert that probabilistic network works.")
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [0]])))
    print(is_same_node(np.array([[2], [0]]), np.array([[1], [2]])))
