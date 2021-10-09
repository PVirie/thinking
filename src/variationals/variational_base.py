import numpy as np
import math

class Model:

    @staticmethod
    def pincer_inference(neighbor_model, estimate_model, s, t):
        var_n = neighbor_model.var
        var_h = estimate_model.var
        inferred_rep = (s * var_h + t * var_n) / (var_n + var_h)

        var = (var_n * var_h) / (var_n + var_h)
        return inferred_rep, 1 / np.sqrt(np.power(2 * np.pi, s.shape[0]) * np.prod(var, axis=0))

    def __init__(self, num_dimensions):
        self.dim = num_dimensions
        self.var = np.ones([self.dim, 1], dtype=np.float32)

    def dims(self):
        return self.dim

    def parameters(self):
        return []

    def compute_entropy(self, x):
        return 0.5 + 0.5 * (np.log(2 * math.pi * self.var))

    def __call__(self, x):
        return self.var


if __name__ == '__main__':

    b = np.random.normal(0, 0.001, [8, 8])

    model = Model(8)
    h = b[:, 0:4]
    v = b[:, 0:1]
    print(model.pincer_inference(model, model, h, v)[0].shape)
