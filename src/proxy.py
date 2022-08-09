import numpy as np
import os
import utilities


class Distinct_item:

    def __init__(self, num_dimensions, candidate_size=None):
        self.dim = num_dimensions
        if candidate_size is not None:
            self.c_size = candidate_size
        else:
            self.c_size = self.dim
        self.C = np.zeros([self.dim, self.c_size], dtype=np.float32)

    def __str__(self):
        return str(self.C)

    def save(self, weight_path):
        if not os.path.exists(weight_path):
            print("Creating directory: {}".format(weight_path))
            os.makedirs(weight_path)
        np.save(os.path.join(weight_path, "C.npy"), self.C)

    def load(self, weight_path):
        if not os.path.exists(weight_path):
            print("Cannot load memories: the path do not exist.")
            return
        self.C = np.load(os.path.join(weight_path, "C.npy"))

    def get_candidates(self):
        return self.C

    def match(self, x):
        return utilities.max_match(x, self.C)

    def store_memory(self, h):
        num_steps = h.shape[1]
        self.C = np.roll(self.C, -num_steps)
        self.C[:, -num_steps:] = h

    def incrementally_learn(self, h):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return

        prop = np.max(self.match(h), axis=0, keepdims=False)
        new_item_mask = prop < 1e-4
        if np.count_nonzero(new_item_mask) > 0:
            self.store_memory(h[:, new_item_mask])
