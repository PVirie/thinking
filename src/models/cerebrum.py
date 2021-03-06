import numpy as np
from energy import Energy_model
import math


class Cerebrum(Energy_model):

    @staticmethod
    def pincer_inference(neighbor_model, estimate_energy_model, s, t):
        _, s_indices, s_prop = neighbor_model.resolve_address(s)
        _, t_indices, t_prop = neighbor_model.resolve_address(t)
        hippocampus_prop = np.power(0.9, t_indices - s_indices) * s_prop * t_prop
        hippocampus_rep = neighbor_model.access_memory(s_indices + 1)

        cortex_potential = neighbor_model.forward_energy(s) + estimate_energy_model.backward_energy(t)
        cortex_pre_prop = 1 / (1 + np.exp(-cortex_potential))
        cortex_prop = np.prod(cortex_pre_prop, axis=0)
        cortex_rep = neighbor_model.enhance(cortex_pre_prop)

        # To do: this is not completely correct. How to enhance the signal with high-level hippocampus?

        return np.where(hippocampus_prop > cortex_prop, hippocampus_rep, cortex_rep)

    def enhance(self, c):
        return np.matmul(self.H, self.resolve_address(c))

    def __init__(self, cortex_dimensions, hippocampus_size, negative_init=False):
        self.c_dim = cortex_dimensions
        self.h_size = hippocampus_size
        self.is_negative_inited = negative_init
        self.C_CH = np.random.normal(0, 0.001, [self.c_dim, self.c_dim + self.h_size]) + (0.0 if not self.is_negative_inited else math.log(1e-8))
        self.H = np.zeros([self.h_size, self.c_dim])  # [oldest, ..., new, newer, newest ]

    def __str__(self):
        if self.is_negative_inited:
            return str((np.transpose(self.C_CH) > math.log(1e-7)).astype(np.int32))
        else:
            return str((np.transpose(self.C_CH) > 0).astype(np.int32))

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

    def compute_energy(self, h, v):
        return np.matmul(np.transpose(v), self.forward_energy(h))

    def forward_energy(self, h, full_rep=False):
        if not full_rep:
            h_, _, __ = self.resolve_address(h)
            h = np.concatenate([h, h_], axis=0)
        return np.matmul(self.C_CH, h)

    def forward(self, h, full_rep=False):
        return 1 / (1 + np.exp(-self.forward_energy(h, full_rep)))

    def backward_energy(self, v):
        return np.matmul(np.transpose(self.C_CH), v)

    def backward(self, v):
        return 1 / (1 + np.exp(-self.backward_energy(v)))

    def incrementally_learn(self, h, v, lr=0.1):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return
        if v.shape[1] == 1 and batch_size > 1:
            v = np.broadcast_to(v, (v.shape[0], batch_size))
        h_, _, __ = self.resolve_address(h)
        hh_ = np.concatenate([h, h_], axis=0)
        self.W = self.W + lr * np.matmul(v - self.forward(hh_, True), np.transpose(hh_)) / batch_size
        # self.W = self.W + lr * np.matmul(v, np.transpose(h - self.backward(v))) / batch_size

        self.store_memory(h)
