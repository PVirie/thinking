import numpy as np


class Pincer_model:

    def __init__(self, neighbor_model, estimate_model):
        self.neighbor_model = neighbor_model
        self.estimate_model = estimate_model

    def inference(self, s, g):
        pincer_potential = self.neighbor_model.__forward_energy(s) + self.estimate_model.__backward_energy(g)
        return np.around(1 / (1 + np.exp(-pincer_potential)))


class Energy_model:

    def __init__(self, num_dimensions):
        self.dim = num_dimensions
        self.W = np.random.normal(0, 0.001, [self.dim, self.dim])
        self.a = np.random.normal(0, 0.001, [self.dim, 1])
        self.b = np.random.normal(0, 0.001, [self.dim, 1])

    def __str__(self):
        return str(np.transpose(self.W))

    def compute_entropy(self, h):
        p = self.__forward(h)
        q = self.__backward(h)
        out_degree = np.sum(-p * np.log(p) - (1 - p) * np.log(1 - p), axis=0)
        in_degree = np.sum(-q * np.log(q) - (1 - q) * np.log(1 - q), axis=0)
        return np.minimum(out_degree, in_degree)

    def __forward_energy(self, h):
        return np.matmul(self.W, h) + self.a

    def __forward(self, h):
        return 1 / (1 + np.exp(-self.__forward_energy(h)))

    def __backward_energy(self, v):
        return np.matmul(np.transpose(self.W), v) + self.b

    def __backward(self, v):
        return 1 / (1 + np.exp(-self.__backward_energy(v)))

    def sample(self, h):
        return np.random.binomial(1, self.__forward(h))

    def incrementally_learn(self, h, v, lr=0.1):
        if v.shape[1] == 1 and h.shape[1] > 1:
            v = np.broadcast_to(v, (v.shape[0], h.shape[1]))
        batch_size = h.shape[1]
        self.W = self.W + lr * np.matmul(v - self.__forward(h), np.transpose(h)) / batch_size
        self.W = self.W + lr * np.matmul(v, np.transpose(h - self.__backward(v))) / batch_size
        self.a = self.a + lr * np.sum(v - self.__forward(h), axis=1, keepdims=True) / batch_size
        self.b = self.b + lr * np.sum(h - self.__backward(v), axis=1, keepdims=True) / batch_size
