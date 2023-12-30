import os
import jax.numpy as jnp
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

        # match all
        p = await self.H.match(x)
        # get neighbors
        kernel = await self.H.roll(1, -1 if forward else 1)

        C = []
        c_prop = []
        
        for i in range(self.candidate_count):
            # find max
            c = await kernel.consolidate(p, use_max=True)

            # keep max
            C.append(c)
            c_prop.append(jnp.max(p, keepdims=False))
            
            # remove max
            m = await kernel.match(c)
            p = p * (1-m)

        return C, c_prop
    

if __name__ == '__main__':
    # To do: test
    pass