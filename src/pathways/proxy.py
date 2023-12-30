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
        
        # this array is use for subtracting selected candidates
        vp = p.copy()
        flatten_p = jnp.reshape(p, [-1])
        for i in range(self.candidate_count):
            # find max
            c = await kernel.consolidate(vp, use_max=True)
            is_null = await c.is_null()
            if not is_null:
                # collect max candidate's prop and candidate
                max_index = jnp.argmax(jnp.reshape(vp, [-1]), keepdims=False)
                c_prop.append(flatten_p[max_index])
                C.append(c)
            
            # remove max
            m = await kernel.match(c, filter_invalid=False)
            vp = vp * (1-m)

        return C, c_prop
    

if __name__ == '__main__':
    # To do: test
    pass