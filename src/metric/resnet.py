import metric_base
from node import Node, Node_tensor_2D
from typing import Sequence, Union, List

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

# class MLP(nn.Module):
#   features: Sequence[int]

#   @nn.compact
#   def __call__(self, x):
#     for feat in self.features[:-1]:
#       x = nn.relu(nn.Dense(feat)(x))
#     x = nn.Dense(self.features[-1])(x)
#     return x

# model = MLP([12, 8, 4])
# key = jax.random.PRNGKey(0)
# batch = jax.random.normal(key, (32, 10))
# variables = model.init(key, batch)
# output = model.apply(variables, batch)
# print(output)

class Model(metric_base.Model, nn.Module):

    def __init__(self):
        pass

    def learn(self, s: Union[Node, List[Node]], t: Node, targets, masks):
        pass

    def distance(self, s: Union[Node, List[Node]], t: Node):
        return 0