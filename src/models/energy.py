import numpy as np

class Energy_model:

    @staticmethod
    def pincer_inference(neighbor_model, estimate_model, s, g):
        var_n = neighbor_model.var
        var_h = estimate_model.var
        inferred_rep = (neighbor_model.forward(s) * var_h + estimate_model.backward(g) * var_n) / (var_n + var_h)

        var = (var_n * var_h) / (var_n + var_h)
        return inferred_rep, 1 / np.sqrt(np.power(2 * np.pi, s.shape[0]) * np.prod(var, axis=0))

    def __init__(self, num_dimensions):
        self.dim = num_dimensions
        self.var = np.zeros([self.dim, 1], dtype=np.float32)
        self.T = np.random.normal(0, 0.001, [self.dim, 1]).astype(np.float32)

    def __str__(self):
        return str(self.var)

    def forward(self, h):
        return h + self.T

    def backward(self, v):
        return v - self.T

    def incrementally_learn(self, h, v, lr=0.01):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return
        if v.shape[1] == 1 and batch_size > 1:
            v = np.broadcast_to(v, (v.shape[0], batch_size))
        delta = v - self.forward(h)
        self.T = self.T + lr * np.sum(delta, axis=1, keepdims=True) / batch_size
        self.var = self.var + lr * (np.sum(delta * delta, axis=1, keepdims=True) / batch_size - self.var)


if __name__ == '__main__':

    b = np.random.normal(0, 0.001, [8, 8])

    model = Energy_model(8)
    h = b[:, 0:4]
    v = b[:, 0:1]
    model.incrementally_learn(h, v)
    print(Energy_model.pincer_inference(model, model, h, v)[0].shape)
