import numpy as np
import os


class Hippocampus:

    def __init__(self, num_dimensions, hippocampus_size, diminishing_factor, candidate_size=None):
        self.dim = num_dimensions
        self.diminishing_factor = diminishing_factor

        self.h_size = hippocampus_size
        self.H = np.zeros([self.dim, self.h_size], dtype=np.float32)  # [oldest, ..., new, newer, newest ]

        if candidate_size is not None:
            self.c_size = candidate_size
        else:
            self.c_size = self.dim
        self.C = np.zeros([self.dim, self.c_size], dtype=np.float32)

        self.positions = np.reshape(np.arange(self.h_size), [-1, 1])

    def __str__(self):
        return str(self.H)

    def save(self, weight_path):
        if not os.path.exists(weight_path):
            print("Creating directory: {}".format(weight_path))
            os.makedirs(weight_path)
        np.save(os.path.join(weight_path, "H.npy"), self.H)
        np.save(os.path.join(weight_path, "C.npy"), self.C)

    def load(self, weight_path):
        if not os.path.exists(weight_path):
            print("Cannot load memories: the path do not exist.")
            return

        self.H = np.load(os.path.join(weight_path, "H.npy"))
        self.C = np.load(os.path.join(weight_path, "C.npy"))

    def match(self, x, match_basis=False):
        H_ = np.transpose(self.H if not match_basis else self.C)
        # match max

        H_ = np.argmax(H_, axis=1, keepdims=True)
        x = np.argmax(x, axis=0, keepdims=True)

        sqr_dist = np.abs(H_ - x)
        prop = np.exp(-0.5 * sqr_dist / 0.1)

        return prop

    def access_memory(self, indices):
        return self.H[:, indices]

    def enhance(self, c):
        prop = self.match(c)
        max_indices = np.argmax(prop, axis=0)
        return self.access_memory(max_indices)

    def resolve_address(self, x, last_indices=None):
        prop = self.match(x)
        if last_indices is not None:
            mask = (self.positions < np.reshape(last_indices, [1, -1])).astype(np.float32)
            prop = prop * mask

        max_indices = np.argmax(prop * self.positions, axis=0)
        supports = np.arange(x.shape[1])
        return max_indices, prop[max_indices, supports]

    def infer(self, s, t):
        t_indices, t_prop = self.resolve_address(t)
        s_indices, s_prop = self.resolve_address(s, t_indices)
        t_prop[t_indices <= s_indices] = 0
        hippocampus_prop = np.power(self.diminishing_factor, t_indices - s_indices - 1) * s_prop * t_prop
        hippocampus_rep = self.access_memory(np.mod(s_indices + 1, self.h_size))
        return hippocampus_rep, hippocampus_prop

    def store_memory(self, h, store_basis=False):
        num_steps = h.shape[1]
        if not store_basis:
            self.H = np.roll(self.H, -num_steps)
            self.H[:, -num_steps:] = h
        else:
            self.C = np.roll(self.C, -num_steps)
            self.C[:, -num_steps:] = h

    def incrementally_learn(self, h):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return

        prop = np.amax(self.match(h, match_basis=True), axis=0, keepdims=False)
        new_item_mask = prop < 1e-2
        if np.count_nonzero(new_item_mask) > 0:
            self.store_memory(h[:, new_item_mask], store_basis=True)
        self.store_memory(h)

    def get_next(self):
        one_step_forwarded = np.roll(self.H, -1)
        return one_step_forwarded

    def get_prev(self):
        one_step_backwarded = np.roll(self.H, 1)
        return one_step_backwarded

    def compute_entropy(self, x):
        prop = self.match(x)
        entropy = np.sum(prop, axis=0, keepdims=False) / self.h_size
        return entropy

    def get_distinct_next_candidate(self, x):
        # x has shape [dim, 1]
        # keep a small set of distinct candidates
        # p(i, 1) = match x to hippocampus
        # q(j, i) = match candidate j to next i + 1
        # candidates' prop = max_i p(i, 1)*q(j, i)

        p = self.match(x)
        q = self.match(self.C)
        q = np.roll(q, -1, axis=0)

        c_prop = np.amax(p * q, axis=0, keepdims=False)
        return self.C, c_prop


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    model = Hippocampus(8, 4, 0.9, 2)
    a = np.random.normal(0, 1, [8, 1])
    b = np.random.normal(0, 1, [8, 1])
    model.incrementally_learn(a)
    model.incrementally_learn(b)
    print("all memories", model)
    a_record = model.access_memory([2, 3])
    print("retrieved", a_record)
    addr = model.resolve_address(a_record)
    print("address", addr)
    prop = model.match(a_record)
    print("match prop", prop)
    entropy = model.compute_entropy(a_record)
    print("entropy", entropy)
    next_records = model.get_next()
    print("next memories", next_records)
    candidates, props = model.get_distinct_next_candidate(a)
    print("candidate props", props)
