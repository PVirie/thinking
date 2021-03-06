import numpy as np
import math


class Hippocampus:

    @staticmethod
    def pincer_inference(neighbor_model, estimate_model, hippocampus, s, t):
        _, s_indices, s_prop = hippocampus.resolve_address(s)
        _, t_indices, t_prop = hippocampus.resolve_address(t)
        hippocampus_prop = np.power(0.9, t_indices - s_indices) * s_prop * t_prop
        hippocampus_rep = hippocampus.access_memory(s_indices + 1)

        cortex_potential = neighbor_model.forward_energy(s) + estimate_model.backward_energy(t)
        cortex_pre_prop = 1 / (1 + np.exp(-cortex_potential))
        cortex_prop = np.prod(cortex_pre_prop, axis=0)
        cortex_rep = hippocampus.enhance(cortex_pre_prop)

        # To do: this is not completely correct. How to enhance the signal with high-level hippocampus?

        return np.where(hippocampus_prop > cortex_prop, hippocampus_rep, cortex_rep)

    def enhance(self, c):
        return np.matmul(self.H, self.resolve_address(c))

    def __init__(self, hippocampus_size):
        self.h_size = hippocampus_size
        self.H = np.zeros([self.h_size, self.c_dim])  # [oldest, ..., new, newer, newest ]

    def __str__(self):
        if self.is_negative_inited:
            return str((np.transpose(self.H) > math.log(1e-7)).astype(np.int32))
        else:
            return str((np.transpose(self.H) > 0).astype(np.int32))

    def resolve_address(self, h):
        prop = np.matmul(self.H, h)
        max_indices = np.argmax(prop, axis=0)
        b = np.zeros((self.H.shape[0], h.shape[1]))
        b[max_indices] = 1
        return b, max_indices, prop[max_indices]

    def store_memory(self, h):
        num_steps = h.shape[1]
        self.H = np.roll(self.H, -num_steps)
        self.H[:, -num_steps:] = h

    def access_memory(self, indices):
        return self.H[:, indices]

    def incrementally_learn(self, h, v, lr=0.1):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return
        self.store_memory(h)


if __name__ == '__main__':
    main()
