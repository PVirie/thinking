import numpy as np
import math
import energy


class Hippocampus:

    def __init__(self, num_dimensions, hippocampus_size):
        self.dim = num_dimensions
        self.h_size = hippocampus_size
        self.H = np.zeros([self.dim, self.h_size])  # [oldest, ..., new, newer, newest ]
        self.diminishing_factor = 0.9

    def pincer_inference(self, neighbor_model, estimate_model, s, t):
        s_indices, s_prop = self.resolve_address(s)
        t_indices, t_prop = self.resolve_address(t)
        hippocampus_prop = np.power(self.diminishing_factor, t_indices - s_indices) * s_prop * t_prop
        hippocampus_rep = self.access_memory(s_indices + 1)

        cortex_rep = energy.Energy_model.pincer_inference(neighbor_model, estimate_model, s, t)

        # To do: this is not completely correct. How to enhance the signal with high-level hippocampus?
        results = np.where(hippocampus_prop > 0.5, hippocampus_rep, cortex_rep)
        return results

    def match(self, x):
        H_ = np.transpose(self.H)
        prop = np.exp(- (np.linalg.norm(x, ord=2, axis=0, keepdims=True)**2 - 2 * np.matmul(H_, x)) + np.linalg.norm(H_, ord=2, axis=1, keepdims=True)**2)
        prop = prop / np.sum(prop, axis=0, keepdims=True)
        return prop

    def compute_entropy(self, x):
        prop = self.match(x)
        entropy = np.sum(prop, axis=0, keepdims=False) / self.h_size
        return entropy

    def enhance(self, c):
        prop = self.match(c)
        max_indices = np.argmax(prop, axis=0)
        return self.access_memory(max_indices)

    def __str__(self):
        return str((self.H > 0).astype(np.int32))

    def resolve_address(self, x):
        prop = self.match(x)
        max_indices = np.argmax(prop, axis=0)
        supports = np.arange(x.shape[1])
        return max_indices, prop[max_indices, supports]

    def store_memory(self, h):
        num_steps = h.shape[1]
        self.H = np.roll(self.H, -num_steps)
        self.H[:, -num_steps:] = h

    def access_memory(self, indices):
        return self.H[:, indices]

    def incrementally_learn(self, h):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return
        self.store_memory(h)


if __name__ == '__main__':
    model = Hippocampus(8, 24)
    model.incrementally_learn(np.array(np.eye(8)))
    model.incrementally_learn(np.array(np.eye(8)))
    print(model)
    a_record = model.access_memory([10, 11])
    print(a_record)
    addr = model.resolve_address(a_record)
    print(addr)
    prop = model.match(a_record)
    print(prop)
