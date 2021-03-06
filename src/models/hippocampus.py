import numpy as np
import math


class Hippocampus:

    def __init__(self, num_dimensions, hippocampus_size):
        self.dim = num_dimensions
        self.h_size = hippocampus_size
        self.H = np.zeros([self.dim, self.h_size])  # [oldest, ..., new, newer, newest ]

    def pincer_inference(self, neighbor_model, estimate_model, s, t):
        _, s_indices, s_prop = self.resolve_address(s)
        _, t_indices, t_prop = self.resolve_address(t)
        hippocampus_prop = np.power(0.9, t_indices - s_indices) * s_prop * t_prop
        hippocampus_rep = self.access_memory(s_indices + 1)

        cortex_potential = neighbor_model.forward_energy(s) + estimate_model.backward_energy(t)
        cortex_pre_prop = 1 / (1 + np.exp(-cortex_potential))
        cortex_rep = self.enhance(cortex_pre_prop)

        # To do: this is not completely correct. How to enhance the signal with high-level hippocampus?
        results = np.where(hippocampus_prop > 0.5, hippocampus_rep, cortex_rep)
        return results

    def enhance(self, c):
        return np.matmul(self.H, self.resolve_address(c)[0])

    def __str__(self):
        return str((self.H > 0).astype(np.int32))

    def resolve_address(self, h):
        prop = np.matmul(np.transpose(self.H), h)
        prop = prop / np.sum(prop, axis=0, keepdims=True)
        max_indices = np.argmax(prop, axis=0)
        b = np.zeros((self.h_size, h.shape[1]))

        supports = np.arange(h.shape[1])
        b[max_indices, supports] = 1
        return b, max_indices, prop[max_indices, supports]

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
