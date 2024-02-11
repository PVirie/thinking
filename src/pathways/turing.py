import os
import jax.numpy as jnp
from jax import jit
from typing import List
from loguru import logger
from base import Pathway, Node


class Model(Pathway):

    def __init__(self, metric_network, diminishing_factor, reach=1, all_pairs=False, name=""):
        self.metric_network = metric_network
        self.diminishing_factor = diminishing_factor
        self.reach = reach  # how many pivots to reach. the larger the coverage, the longer the training.
        self.no_pivot = all_pairs  # extreme training condition all node to all nodes

        self.name = name
        self.learn_steps = 0


    def save(self, path):
        weight_path = os.path.join(path, "turing")
        os.makedirs(weight_path, exist_ok=True)


    def load(self, path):
        weight_path = os.path.join(path, "turing")


    async def infer(self, s: Node, t: Node):
        pass


    async def incrementally_learn(self, path: List[Node], pivots):
        pass




if __name__ == '__main__':
    pass
