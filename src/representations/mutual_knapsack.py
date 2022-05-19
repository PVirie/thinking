import os
import rep_base
import numpy as np
import sys


class Model(rep_base.Model):

    def __init__(self, knapsack_size):
        self.data = np.zeros([knapsack_size], dtype=np.int32)

    def set(self, v):
        self.data[:] = v

    def dims(self):
        return self.data.shape[0]

    def __dist__(self, c):
        return np.sum(c.data != self.data)

    def dist(self, c):
        if c.dims() != self.dims():
            return None
        return self.__dist__(c)

    def sub_dist(self, c):
        if c.dims() != self.dims():
            return None

        if np.sum((c.data != 0) * (self.data - c.data)) != 0:
            return np.inf

        return self.__dist__(c)

    def min_dist(self, cs):
        # cs is a list of knapsacks
        tobestacked = []
        for element in cs:
            tobestacked.append(element.data)
        matched = (np.stack(tobestacked, axis=0) == np.reshape(self.data, [1, self.data.shape[0]]))
        return self.data.shape[0] - np.sum(np.amax(matched, axis=0))


if __name__ == '__main__':
    x = Model(8)
    y = Model(8)
    z = Model(8)

    x.set([1, 2, 3, 4, 5, 6, 7, 8])
    y.set([1, 2, 3, 0, 5, 6, 7, 8])
    z.set([1, 2, 0, 4, 5, 6, 7, 8])

    print(x.dist(y))

    print(x.sub_dist(y))
    print(x.sub_dist(z))

    print(z.min_dist([x, y]))
    print(x.min_dist([y, z]))
    print(y.min_dist([z]))
