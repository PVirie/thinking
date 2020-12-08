import numpy as np
import math



class Energy_model:

    @staticmethod
    def pincer_inference(neighbor_model, estimate_model, s, g):
        pincer_potential = neighbor_model.forward_energy(s) + estimate_model.backward_energy(g)
        return 1 / (1 + np.exp(-pincer_potential))

    @staticmethod
    def enhance(c):
        b = np.zeros((c.shape[0], c.shape[1]))
        b[np.argmax(c, axis=0), np.arange(c.shape[1])] = 1
        return b

    def __init__(self, num_dimensions, negative_init=False):
        self.dim = num_dimensions
        self.is_negative_inited = negative_init
        self.W = np.random.normal(0, 0.001, [self.dim, self.dim]) + (0.0 if not self.is_negative_inited else math.log(1e-8))

    def __str__(self):
        if self.is_negative_inited:
            return str((np.transpose(self.W) > math.log(1e-7)).astype(np.int32))
        else:
            return str((np.transpose(self.W) > 0).astype(np.int32))

    def compute_entropy(self, h):
        p = self.forward(h)
        q = self.backward(h)
        out_degree = np.sum(-p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8), axis=0)
        in_degree = np.sum(-q * np.log(q + 1e-8) - (1 - q) * np.log(1 - q + 1e-8), axis=0)
        return np.minimum(out_degree, in_degree)

    def compute_prop(self, h, v):
        '''
        v \in {0, 1}^{num_dimensions}
        '''
        p = self.forward(h)
        return np.prod(v * p + (1 - v) * (1 - p), axis=0)

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

    def sample(self, h):
        return np.random.binomial(1, self.forward(h))

    def incrementally_learn(self, h, v, lr=0.1):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return
        if v.shape[1] == 1 and batch_size > 1:
            v = np.broadcast_to(v, (v.shape[0], batch_size))
        self.W = self.W + lr * np.matmul(v - self.forward(h), np.transpose(h)) / batch_size
        # self.W = self.W + lr * np.matmul(v, np.transpose(h - self.backward(v))) / batch_size

    def learn(self, h, v, target_prop, lr=0.9, steps=10):
        if v.shape[1] == 1 and h.shape[1] > 1:
            v = np.broadcast_to(v, (v.shape[0], h.shape[1]))
        batch_size = h.shape[1]
        log_target = np.log(target_prop + 1e-8)
        for i in range(steps):
            # forward_delta = np.maximum(log_target - np.sum(v * self.forward_energy(h), axis=0), 0.0)
            backward_delta = np.maximum(log_target - np.sum(h * self.backward_energy(v), axis=0), 0.0)

            # self.W = self.W + lr * np.matmul(v, np.transpose(forward_delta * h)) / batch_size
            self.W = self.W + lr * np.matmul(backward_delta * v, np.transpose(h))


if __name__ == '__main__':

    b = np.zeros((8, 8))
    b[np.arange(8), np.arange(8)] = 1

    model = Energy_model(8, negative_init=True)
    h = b[:, 0:4]
    v = b[:, 0:1]
    target = np.array([0.02, 0.03, 0.04, 0.05])
    model.learn(h, v, target)
    print(model.W)
    print(np.exp(model.compute_energy(h, v)))
