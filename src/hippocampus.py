import numpy as np
import os
import proxy


class Hippocampus:

    def __init__(self, num_dimensions, memory_size, chunk_size, diminishing_factor, candidate_size=None):
        self.dim = num_dimensions
        self.diminishing_factor = diminishing_factor

        self.chunk_size = chunk_size
        self.h_size = memory_size
        self.H = np.zeros([self.dim, self.h_size, self.chunk_size], dtype=np.float32)  # [oldest, ..., new, newer, newest ]
        self.flat_H = np.reshape(self.H, [self.dim, -1])

        self.bases = proxy.Distinct_item(self.dim, candidate_size)

    def __str__(self):
        return str(self.H)

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
        self.bases.load(weight_path)

    def match(self, x):
        H_ = np.transpose(self.flat_H)
        # match max

        H_ = np.argmax(H_, axis=1, keepdims=True).astype(np.float32)
        x = np.argmax(x, axis=0, keepdims=True).astype(np.float32)

        sqr_dist = np.abs(H_ - x)
        prop = np.exp(-0.5 * sqr_dist / 0.1)

        return prop

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
        hippocampus_rep = self.access_memory(np.mod(s_best_indices + 1, self.h_size))

        hippocampus_prop = np.reshape(hippocampus_prop, [1, -1])
        return hippocampus_rep, hippocampus_prop

    def store_memory(self, h):
        num_steps = h.shape[1]
        self.H = np.roll(self.H, -1, axis=1)
        self.H[:, self.h_size - 1, :num_steps] = h
        self.flat_H = np.reshape(self.H, [self.dim, -1])

    def incrementally_learn(self, h):
        batch_size = h.shape[1]
        if batch_size <= 0:
            return

        self.store_memory(h)
        self.bases.incrementally_learn(h)

    # def get_next(self):
    #     one_step_forwarded = np.roll(self.H, -1)
    #     return one_step_forwarded

    # def get_prev(self):
    #     one_step_backwarded = np.roll(self.H, 1)
    #     return one_step_backwarded

    def compute_entropy(self, x):
        prop = self.match(x)
        entropy = np.sum(prop, axis=0, keepdims=False) / (self.h_size * self.chunk_size)
        return entropy

    def get_distinct_next_candidate(self, x):
        # x has shape [dim, 1]
        # keep a small set of distinct candidates
        # p(i, 1) = match x to hippocampus
        # q(j, i) = match candidate j to next i + 1
        # candidates' prop = max_i p(i, 1)*q(j, i)

        C = self.bases.get_candidates()

        p = self.match(x)
        q = self.match(C)
        # the last element of each chunk should be ignored
        q = np.roll(q, -1, axis=0)

        c_prop = np.amax(p * q, axis=0, keepdims=False)
        return C, c_prop


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    model = Hippocampus(8, 4, 5, 0.9)
    a = np.random.normal(0, 1, [8, 2])
    b = np.random.normal(0, 1, [8, 3])
    model.incrementally_learn(a)
    model.incrementally_learn(b)
    print("all memories", model)

    rep, prop = model.infer(b[:, 0:1], b[:, 2:3])
    print(prop)

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
