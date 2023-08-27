import sys
import os
import math
import numpy as np
import asyncio
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
        np.save(os.path.join(weight_path, "H.npy"), self.H.data)

    def load(self, path):
        weight_path = os.path.join(path, "hippocampus")
        self.H.data = np.load(os.path.join(weight_path, "H.npy"))

    async def enhance(self, c: Node):
        # flatten self.H, preserve indices
        prop = await self.H.match(c)

        # find max row and column
        max_indices = np.argmax(prop, axis=1)
        max_prop = np.max(prop, axis=1)
        # find max row
        max_row = np.argmax(max_prop, axis=0)
        # find max column
        max_col = max_indices[max_row]
        return await self.H.access(max_row, max_col)

    async def infer(self, s: Node, t: Node):
        s_prop = await self.H.match(s)
        t_prop = await self.H.match(t)

        s_max_indices = np.argmax(s_prop, axis=1)
        t_max_indices = np.argmax(t_prop, axis=1)
        s_max = np.max(s_prop, axis=1)
        t_max = np.max(t_prop, axis=1)

        causality = (t_max_indices > s_max_indices).astype(np.float32)
        # flip to take the last occurrence
        best = (self.h_size - 1) - np.argmax(np.flip(s_max * t_max * causality, axis=0), axis=0)

        s_best_index = s_max_indices[best]
        t_best_index = t_max_indices[best]
        s_best_prop = s_max[best]
        t_best_prop = t_max[best]

        # print(best, s_best_indices, t_best_indices)

        if s_best_index >= self.chunk_size:
            return None, None

        hippocampus_prop = pow(self.diminishing_factor, t_best_index - s_best_index - 1) * s_best_prop * t_best_prop
        hippocampus_rep = await self.H.access(best, s_best_index + 1)
        
        return hippocampus_rep, hippocampus_prop


    async def incrementally_learn(self, hs: List[Node]):
        await self.H.append(hs)

        
    async def compute_entropy(self, xs: List[Node]):
        entropies = []
        for x in xs:
            prop = await self.H.match(x)
            entropies.append(np.mean(prop))
        return entropies


    async def sample(self, x: Node, forward=True):
        # To do normalize distinct?
        prop = await self.H.match(x)
        res = await self.H.roll(1, -1 if forward else 1)
        return await res.consolidate(prop)



if __name__ == '__main__':
    pass
    # To do: write test
