import sys
import os
import math
import numpy as np
import asyncio
from typing import List
from loguru import logger
from base import Pathway
import hippocampus

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..'))

from metric import Node, Node_tensor_2D


class Model(hippocampus.Model):

    def __init__(self, memory_size, chunk_size, candidate_count):
        # call super method
        super().__init__(memory_size, chunk_size, 1.0)
        self.candidate_count = candidate_count


    def incrementally_learn(self, hs: List[Node]):
        super().incrementally_learn(hs)


    async def get_candidates(self, x: Node, forward=True):

        C = []
        c_prop = np.zeros(self.candidate_count, dtype=np.float32)
        p = await self.H.match(x)

        if forward:
            kernel = self.H.roll(1, -1)
        else:
            kernel = self.H.roll(1, 1)

        for i in range(self.candidate_count):
            c = kernel.consolidate(p, use_max=True)
            C.append(c)
            c_prop[i] = np.max(p, axis=0, keepdims=False)
            m = kernel.match(c)
            p = p * (1-m)

        return C, c_prop
    

if __name__ == '__main__':
    # To do: test
    pass