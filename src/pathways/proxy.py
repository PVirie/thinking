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

    def incrementally_learn(self, h):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return

        prop = np.max(self.match(h), axis=0, keepdims=False)
        new_item_mask = prop < 1e-4
        if np.count_nonzero(new_item_mask) > 0:
            self.store_memory(h[:, new_item_mask])


    def get_candidates(self):
        return self.C

    def match(self, x):
        return utilities.max_match(x, self.C)

    def store_memory(self, h):
        num_steps = h.shape[1]
        self.C = np.roll(self.C, -num_steps)
        self.C[:, -num_steps:] = h

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