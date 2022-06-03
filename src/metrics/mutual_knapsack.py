import metric_base
import torch
import torch.nn as nn


class Model(metric_base.Model):

    def __init__(self, dims):
        self.dims = dims

    def dist(self, a, b):
        # a and b are both tensors of the following shape: [batch, dim, bandwidth]
        # return [batch]
        signed_dist = a - b
        return torch.sum(torch.sum(torch.abs(signed_dist), dim=2), dim=1) / 2

    def represent(self, a):
        return torch.argmax(a, dim=2)


if __name__ == '__main__':
    import numpy as np
    metric = Model()

    a = torch.from_numpy(np.random.normal(0, 1.0, [1, 8, 4]))
    b = torch.from_numpy(np.random.normal(0, 1.0, [1, 8, 4]))

    print(a, b)
    dist = metric.dist(a, b)
    print(dist)
    dist = metric.dist(a, a)
    print(dist)
    dist = metric.dist(a, -a)
    print(dist)
