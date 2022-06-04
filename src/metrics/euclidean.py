import metric_base
import torch
import torch.nn as nn


class Model(metric_base.Model):

    def __init__(self, dims):
        self.dims = dims

    def sqr_dist(self, a, b):
        # a and b are both tensors of the following shape: [dim, length, batch]
        # return [dim, length, batch]
        signed_dist = a - b
        return torch.square(signed_dist)

    def represent(self, a):
        return a


if __name__ == '__main__':
    import numpy as np
    metric = Model(8)

    a = torch.from_numpy(np.random.normal(0, 1.0, [8, 4, 1]))
    b = torch.from_numpy(np.random.normal(0, 1.0, [8, 4, 1]))

    print(a, b)
    dist = metric.sqr_dist(a, b)
    print(dist)
    dist = metric.sqr_dist(a, a)
    print(dist)
    dist = metric.sqr_dist(a, -a)
    print(dist)
