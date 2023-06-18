import math
import os
import numpy as np
import asyncio
from typing import List
from node import Node, Node_tensor_2D
from loguru import logger


class Model:

    def __init__(self, memory_size, chunk_size, diminishing_factor):
        self.diminishing_factor = diminishing_factor

        self.chunk_size = chunk_size
        self.h_size = memory_size
        self.H = Node_tensor_2D(self.h_size, self.chunk_size)  # [[oldest, ..., new, newer, newest ], ...]
        
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
        hippocampus_rep = self.H.access(best, s_best_index + 1)
        
        return hippocampus_rep, hippocampus_prop


    async def incrementally_learn(self, hs: List[Node]):
        self.H.append(hs)

        
    async def compute_entropy(self, x: Node):
        prop = await self.H.match(x)
        entropy = np.mean(prop)
        return entropy


    async def sample(self, x: Node, forward=True):
        # To do normalize distinct?
        prop = await self.H.match(x)
        if forward:
            return self.H.roll(1, -1).consolidate(prop)
        else:
            return self.H.roll(1, 1).consolidate(prop)



if __name__ == '__main__':
    pass
    # To do: write test
