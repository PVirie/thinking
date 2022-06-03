import numpy as np


class Hippocampus:

    def __init__(self, num_dimensions, hippocampus_size):
        self.dim = num_dimensions
        self.h_size = hippocampus_size
        self.H = np.zeros([self.dim, self.h_size], dtype=np.float32)  # [oldest, ..., new, newer, newest ]
        self.diminishing_factor = 0.9

    def infer(self, s, t):
        s_indices, s_prop = self.resolve_address(s)
        t_indices, t_prop = self.resolve_address(t, s_indices)
        t_prop[t_indices <= s_indices] = 0
        hippocampus_prop = np.power(self.diminishing_factor, t_indices - s_indices) * s_prop * t_prop
        hippocampus_rep = self.access_memory(np.mod(s_indices + 1, self.h_size))
        return hippocampus_rep, hippocampus_prop

    def match(self, x):
        H_ = np.transpose(self.H)
        # use isometric gaussian now, should be using the metric in the neighbor model for computing entropy.
        prop = np.exp(- (np.linalg.norm(H_, ord=2, axis=1, keepdims=True)**2 - 2 * np.matmul(H_, x) + np.linalg.norm(x, ord=2, axis=0, keepdims=True)**2))
        return prop

    def compute_entropy(self, x):
        prop = self.match(x)
        norm_prop = prop / np.sum(prop, axis=0, keepdims=True)
        entropy = np.sum(norm_prop, axis=0, keepdims=False) / self.h_size
        return entropy

    def enhance(self, c):
        prop = self.match(c)
        max_indices = np.argmax(prop, axis=0)
        return self.access_memory(max_indices)

    def get_next(self):
        one_step_forwarded = np.roll(self.H, -1)
        return one_step_forwarded

    def __str__(self):
        return str(self.H)

    def resolve_address(self, x, start_indices=None):
        prop = self.match(x)
        if start_indices is not None:
            positions = np.reshape(np.arange(self.h_size), [-1, 1])
            mask = (positions > np.reshape(start_indices, [1, -1])).astype(np.float32)
            prop = prop * mask

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
    np.set_printoptions(precision=2)
    model = Hippocampus(8, 4)
    a = np.random.normal(0, 1, [8, 1])
    b = np.random.normal(0, 1, [8, 1])
    model.incrementally_learn(a)
    model.incrementally_learn(b)
    print(model)
    a_record = model.access_memory([2, 3])
    print(a_record)
    addr = model.resolve_address(a_record)
    print(addr)
    prop = model.match(a_record)
    print(prop)
    entropy = model.compute_entropy(a_record)
    print(entropy)
    next_records = model.get_next()
    print(next_records)
