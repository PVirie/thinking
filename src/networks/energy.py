import numpy as np


def pincer_inference(neighbor_model, estimate_model, s, g):
    pincer_potential = neighbor_model.forward_energy(s) + estimate_model.backward_energy(g)
    return 1 / (1 + np.exp(-pincer_potential))


class Energy_model:

    def __init__(self, num_dimensions):
        self.dim = num_dimensions
        self.W = np.random.normal(0, 0.001, [self.dim, self.dim])
        self.a = np.random.normal(0, 0.001, [self.dim, 1])
        self.b = np.random.normal(0, 0.001, [self.dim, 1])

    def __str__(self):
        return str((np.transpose(self.W) > 0).astype(np.int32))

    def compute_entropy(self, h):
        p = self.forward(h)
        q = self.backward(h)
        out_degree = np.sum(-p * np.log(p) - (1 - p) * np.log(1 - p), axis=0)
        in_degree = np.sum(-q * np.log(q) - (1 - q) * np.log(1 - q), axis=0)
        return np.minimum(out_degree, in_degree)

    def compute_prop(self, h, v):
        '''
        v \in {0, 1}^{num_dimensions}
        '''
        p = self.forward(h)
        return np.prod(v*p + (1-v)*(1-p), axis=0)

    def forward_energy(self, h):
        return np.matmul(self.W, h) + self.a

    def forward(self, h):
        return 1 / (1 + np.exp(-self.forward_energy(h)))

    def backward_energy(self, v):
        return np.matmul(np.transpose(self.W), v) + self.b

    def backward(self, v):
        return 1 / (1 + np.exp(-self.backward_energy(v)))

    def sample(self, h):
        return np.random.binomial(1, self.forward(h))

    def incrementally_learn(self, h, v, lr=0.1):
        if v.shape[1] == 1 and h.shape[1] > 1:
            v = np.broadcast_to(v, (v.shape[0], h.shape[1]))
        batch_size = h.shape[1]
        self.W = self.W + lr * np.matmul(v - self.forward(h), np.transpose(h)) / batch_size
        self.W = self.W + lr * np.matmul(v, np.transpose(h - self.backward(v))) / batch_size
        self.a = self.a + lr * np.sum(v - self.forward(h), axis=1, keepdims=True) / batch_size
        self.b = self.b + lr * np.sum(h - self.backward(v), axis=1, keepdims=True) / batch_size
