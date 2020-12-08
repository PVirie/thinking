import numpy as np
import energy


class Cerebrum(Energy_model):

    @staticmethod
    def pincer_inference(neighbor_model, estimate_model, s, g):
        pincer_potential = neighbor_model.forward_energy(s) + estimate_model.backward_energy(g)
        return 1 / (1 + np.exp(-pincer_potential))

    @staticmethod
    def enhance(c):
        b = np.zeros((c.shape[0], c.shape[1]))
        b[np.argmax(c, axis=0), np.arange(c.shape[1])] = 1
        return b

    def __init__(self, cortex_dimensions, hippocampus_dimensions, negative_init=False):
        self.c_dim = cortex_dimensions
        self.h_dim = hippocampus_dimensions
        self.is_negative_inited = negative_init
        self.CH_C = np.random.normal(0, 0.001, [self.c_dim + self.h_dim, self.c_dim]) + (0.0 if not self.is_negative_inited else math.log(1e-8))
        self.H_H = np.random.normal(0, 0.001, [self.h_dim, self.h_dim]) + (0.0 if not self.is_negative_inited else math.log(1e-8))


    def __str__(self):
        if self.is_negative_inited:
            return str((np.transpose(self.CH_C) > math.log(1e-7)).astype(np.int32))
        else:
            return str((np.transpose(self.CH_C) > 0).astype(np.int32))

    def compute_energy(self, h, v):
        return np.matmul(np.transpose(v), np.matmul(self.W, h))

    def forward_energy(self, h):
        return np.matmul(self.W, h)

    def forward(self, h):
        return 1 / (1 + np.exp(-self.forward_energy(h)))

    def backward_energy(self, v):
        return np.matmul(np.transpose(self.W), v)

    def backward(self, v):
        return 1 / (1 + np.exp(-self.backward_energy(v)))

    def incrementally_learn(self, h, v, lr=0.1):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return
        if v.shape[1] == 1 and batch_size > 1:
            v = np.broadcast_to(v, (v.shape[0], batch_size))
        self.W = self.W + lr * np.matmul(v - self.forward(h), np.transpose(h)) / batch_size
        # self.W = self.W + lr * np.matmul(v, np.transpose(h - self.backward(v))) / batch_size
