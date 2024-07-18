import os
import jax.numpy as jnp
from typing import List
from loguru import logger
from base import Pathway, Node, Node_tensor_2D


class Model(Pathway):

    def __init__(self, memory_size, chunk_size, diminishing_factor, embedding_dim):
        self.diminishing_factor = diminishing_factor

        self.chunk_size = chunk_size
        self.h_size = memory_size
        self.H = Node_tensor_2D(self.h_size, self.chunk_size, node_dim=embedding_dim)  # [[oldest, ..., new, newer, newest ], ...]
        

    def save(self, path):
        weight_path = os.path.join(path, "hippocampus")
        os.makedirs(weight_path, exist_ok=True)
        # self.H.data is an np array
        jnp.save(os.path.join(weight_path, "H.npy"), self.H.data)

    def load(self, path):
        weight_path = os.path.join(path, "hippocampus")
        self.H.data = jnp.load(os.path.join(weight_path, "H.npy"))

    async def enhance(self, c: Node):
        # flatten self.H, preserve indices
        prop = await self.H.match(c)

        # find max row and column
        max_indices = jnp.argmax(prop, axis=1)
        max_prop = jnp.max(prop, axis=1)
        # find max row
        max_row = jnp.argmax(max_prop, axis=0)
        # find max column
        max_col = max_indices[max_row]
        return await self.H.access(max_row, max_col)

    async def infer(self, s: Node, t: Node):
        # find the latest distance between s and t
        s_prop = await self.H.match(s)
        t_prop = await self.H.match(t)

        s_max_indices = jnp.argmax(s_prop, axis=1)
        t_max_indices = jnp.argmax(t_prop, axis=1)
        s_max = jnp.max(s_prop, axis=1)
        t_max = jnp.max(t_prop, axis=1)

        causality = (t_max_indices > s_max_indices).astype(jnp.float32)
        # flip to take the last occurrence
        best = (self.h_size - 1) - jnp.argmax(jnp.flip(s_max * t_max * causality, axis=0), axis=0)

        s_best_index = s_max_indices[best]
        t_best_index = t_max_indices[best]
        s_best_prop = s_max[best]
        t_best_prop = t_max[best]

        # print(best, s_best_indices, t_best_indices)

        if s_best_index >= self.chunk_size:
            return None, None

        hippocampus_prop = pow(self.diminishing_factor, t_best_index - s_best_index) * s_best_prop * t_best_prop
        hippocampus_rep = await self.H.access(best, s_best_index + 1)
        
        return hippocampus_rep, hippocampus_prop


    async def incrementally_learn(self, hs: List[Node]):
        await self.H.append(hs)

        
    async def compute_entropy(self, xs: List[Node]):
        entropies = []
        for x in xs:
            prop = await self.H.match(x)
            entropies.append(jnp.mean(prop))
        return entropies


    async def sample(self, x: Node, forward=True):
        # To do normalize distinct?
        prop = await self.H.match(x)
        res = await self.H.roll(1, -1 if forward else 1)
        return await res.consolidate(prop)


    async def distance(self, s: List[Node], t: List[Node]):
        # find the minimal distance between all pairs of s and t
        # return a matrix of shape [len(t), len(s)]
        s_prop = await self.H.match_many(s)
        t_prop = await self.H.match_many(t)

        # s_prop has shape [len(s), max_row, max_col]
        # t_prop has shape [len(t), max_row, max_col]
        s_max_indices = jnp.argmax(s_prop, axis=2)
        t_max_indices = jnp.argmax(t_prop, axis=2)
        s_max = jnp.max(s_prop, axis=2)
        t_max = jnp.max(t_prop, axis=2)

        # s_max_* has shape [len(s), max_row]
        # t_max_* has shape [len(t), max_row]
        # now make s_max_indices and t_max_indices have shape [len(t), len(s), max_row]
        s_max_indices = jnp.expand_dims(s_max_indices, axis=0)
        s_max_indices = jnp.repeat(s_max_indices, len(t), axis=0)
        t_max_indices = jnp.expand_dims(t_max_indices, axis=1)
        t_max_indices = jnp.repeat(t_max_indices, len(s), axis=1)
        s_max = jnp.expand_dims(s_max, axis=0)
        t_max = jnp.expand_dims(t_max, axis=1)

        causality = (t_max_indices >= s_max_indices).astype(jnp.float32)
        dist_score = self.chunk_size - (t_max_indices - s_max_indices)
        # choose the smallest distance
        logit = s_max * t_max * causality * dist_score
        best = jnp.argmax(logit, axis=2)
        best_value = jnp.max(logit, axis=2)

        # best has shape [len(t), len(s)]
        t_indexer, s_indexer = jnp.meshgrid(jnp.arange(len(t)), jnp.arange(len(s)), indexing='ij')
        # get s_best_index and t_best_index of shape [len(t), len(s)]
        s_best_index = s_max_indices[t_indexer, s_indexer, best]
        t_best_index = t_max_indices[t_indexer, s_indexer, best]

        distances = pow(self.diminishing_factor, t_best_index - s_best_index)

        # filter distance where best is invalid
        distances = distances * (best_value > 0)
        return distances



if __name__ == '__main__':
    # test
    import asyncio
    import random

    model = Model(16, 8, 0.9, 4)
    rep = [Node(r) for r in jnp.eye(4, dtype=jnp.float32)]

    async def test():
        await model.incrementally_learn([rep[i] for i in [0, 1, 2, 3]])
        await model.incrementally_learn([rep[i] for i in [0, 2, 1, 3]])
        
        # one pair lastest distance
        _, prop = await model.infer(rep[2], rep[3])
        print(prop)

        # all pair minimal distance
        all_pair_distances = await model.distance(rep, rep)
        print(all_pair_distances)

    asyncio.run(test())