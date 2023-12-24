import sys
import os
import math
import jax.numpy as jnp
import asyncio
from typing import List
from loguru import logger
from base import Pathway, Node, Node_tensor_2D
import hippocampus


class Model(hippocampus.Model):

    def __init__(self, memory_size, chunk_size, candidate_count, embedding_dim):
        # call super method
        super().__init__(memory_size, chunk_size, 1.0, embedding_dim)
        self.candidate_count = candidate_count


    def save(self, path):
        weight_path = os.path.join(path, "proxy")
        os.makedirs(weight_path, exist_ok=True)
        # self.H.data is an np array
        jnp.save(os.path.join(weight_path, "H.npy"), self.H.data)

    def load(self, path):
        weight_path = os.path.join(path, "proxy")
        self.H.data = jnp.load(os.path.join(weight_path, "H.npy"))

        
    async def incrementally_learn(self, hs: List[Node]):
        await super().incrementally_learn(hs)


    async def get_candidates(self, x: Node, forward=True):

        p = await self.H.match(x)
        kernel = await self.H.roll(1, -1 if forward else 1)
        # pick top-distinct candidate_count from kernel, can't simply use top-k need to be distinct
        # for now return all candidates
        return kernel, p

        # C = []
        # c_prop = jnp.zeros(self.candidate_count, dtype=jnp.float32)
        
        # for i in range(self.candidate_count):
        #     c = await kernel.consolidate(p, use_max=True)
        #     C.append(c)
        #     c_prop[i] = jnp.max(p, keepdims=False)
        #     m = await kernel.match(c)
        #     p = p * (1-m)

        # return C, c_prop
    

if __name__ == '__main__':
    # To do: test
    pass