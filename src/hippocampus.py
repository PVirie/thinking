import numpy as np
import os
import proxy
import utilities
import asyncio
from loguru import logger


class Model:

    def __init__(self, num_dimensions, memory_size, chunk_size, diminishing_factor, candidate_size=None):
        self.dim = num_dimensions
        self.diminishing_factor = diminishing_factor

        self.chunk_size = chunk_size
        self.h_size = memory_size
        self.H = np.zeros([self.dim, self.h_size, self.chunk_size], dtype=np.float32)  # [oldest, ..., new, newer, newest ]
        self.flat_H = np.reshape(self.H, [self.dim, -1])

        self.bases = proxy.Distinct_item(self.dim, candidate_size)

    def __str__(self):
        line = ""
        for r in range(self.H.shape[1]):
            for c in range(self.H.shape[2]):
                line += str(np.argmax(self.H[:, r, c])) + " "
            line += "\n"
        return line

    def save(self, weight_path):
        if not os.path.exists(weight_path):
            print("Creating directory: {}".format(weight_path))
            os.makedirs(weight_path)
        np.save(os.path.join(weight_path, "H.npy"), self.H)
        self.bases.save(weight_path)

    def load(self, weight_path):
        if not os.path.exists(weight_path):
            print("Cannot load memories: the path do not exist.")
            return

        self.H = np.load(os.path.join(weight_path, "H.npy"))
        self.flat_H = np.reshape(self.H, [self.dim, -1])
        self.bases.load(weight_path)

    def match(self, x):
        return utilities.max_match(x, self.flat_H)

    def access_memory(self, indices):
        return self.flat_H[:, indices]

    def enhance(self, c):
        prop = self.match(c)
        max_indices = np.argmax(prop, axis=0)
        return self.access_memory(max_indices)

    def infer(self, s, t):
        s_prop = np.reshape(self.match(s), [self.h_size, self.chunk_size, -1])
        t_prop = np.reshape(self.match(t), [self.h_size, self.chunk_size, -1])

        s_max_indices = np.argmax(s_prop, axis=1)
        t_max_indices = np.argmax(t_prop, axis=1)
        s_max = np.max(s_prop, axis=1)
        t_max = np.max(t_prop, axis=1)

        causality = (t_max_indices > s_max_indices).astype(np.float32)
        best = (self.h_size - 1) - np.argmax(np.flip(s_max * t_max * causality, axis=0), axis=0)

        batcher = np.arange(s.shape[1])
        s_best_indices = s_max_indices[best, batcher]
        t_best_indices = t_max_indices[best, batcher]
        s_best_prop = s_max[best, batcher]
        t_best_prop = t_max[best, batcher]

        # print(best, s_best_indices, t_best_indices)

        hippocampus_prop = np.power(self.diminishing_factor, t_best_indices - s_best_indices - 1) * s_best_prop * t_best_prop
        hippocampus_rep = self.access_memory(best * self.chunk_size + np.mod(s_best_indices + 1, self.chunk_size))

        hippocampus_prop = np.reshape(hippocampus_prop, [-1])
        return hippocampus_rep, hippocampus_prop

    def store_memory(self, h):
        num_steps = h.shape[1]
        self.H = np.roll(self.H, -1, axis=1)
        self.H[:, self.h_size - 1, :num_steps] = h
        self.H[:, self.h_size - 1, num_steps:] = 0
        self.flat_H = np.reshape(self.H, [self.dim, -1])

    def incrementally_learn(self, h):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return

        self.store_memory(h)
        self.bases.incrementally_learn(h)

    def compute_entropy(self, x):
        prop = self.match(x)
        entropy = np.sum(prop, axis=0, keepdims=False) / (self.h_size * self.chunk_size)
        return entropy

    def get_distinct_next_candidate(self, x, forward=True):
        # x has shape [dim, 1]
        # keep a small set of distinct candidates
        # p(i, 1) = match x to hippocampus
        # q(j, i) = match candidate j to next i + 1
        # candidates' prop = max_i p(i, 1)*q(j, i)

        C = self.bases.get_candidates()
        p = self.match(x)
        q = self.match(C)

        q = np.reshape(q, [self.h_size, self.chunk_size, C.shape[1]])
        if forward:
            # the last element of each chunk should be ignored
            q = np.roll(q, -1, axis=1)
            q[:, -1, :] = 0
        else:
            # the first element of each chunk should be ignored
            q = np.roll(q, 1, axis=1)
            q[:, 0, :] = 0
        q = np.reshape(q, [self.h_size * self.chunk_size, C.shape[1]])

        c_prop = np.max(p * q, axis=0, keepdims=False)

        return C, c_prop


if __name__ == '__main__':

    representations = utilities.generate_onehot_representation(np.arange(8), 8)

    np.set_printoptions(precision=2)
    model = Model(8, 4, 6, 0.9)
    a = representations[:, [0, 1, 2]]
    b = representations[:, [3, 4, 5, 6]]
    c = representations[:, [7, 0, 3, 4, 0, 2]]
    model.incrementally_learn(a)
    model.incrementally_learn(b)
    model.incrementally_learn(c)
    print("all memories", model)

    C, c_prop = model.get_distinct_next_candidate(representations[:, 0:1])
    print([(np.argmax(C[:, i]), c_prop[i]) for i in range(8)])

    # rep, prop = model.infer(b[:, 0:1], b[:, 2:3])
    # print(prop)

    # prop = model.match(a)
    # print("match prop", prop)

    # a_record = model.access_memory([2, 3])
    # print("retrieved", a_record)
    # addr = model.resolve_address(a_record)
    # print("address", addr)
    # entropy = model.compute_entropy(a_record)
    # print("entropy", entropy)
    # next_records = model.get_next()
    # print("next memories", next_records)
    # candidates, props = model.get_distinct_next_candidate(a)
    # print("candidate props", props)
