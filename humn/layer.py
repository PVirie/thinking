from .interface import *
from typing import List, Tuple

class Layer:
    def __init__(self, name, heuristics, hippocampus, proxy):
        self.name = name
        self.heuristics = heuristics
        self.hippocampus = hippocampus
        self.proxy = proxy


    def save(self, weight_path):
        pass

    def load(self, weight_path):
        pass


    async def incrementally_learn(self, path: State_Sequence, pivots_indices):
        if len(path) < 2:
            return

        await self.hippocampus.incrementally_learn(path)
        await self.proxy.incrementally_learn(path)

        # await self.heuristics.incrementally_learn(path, pivots_indices)
        hippocampus_distances = await self.hippocampus.distance(path, [path[i] for i in pivots_indices])
        await self.heuristics.incrementally_learn_2(path, pivots_indices, hippocampus_distances)


    async def infer_next_step(self, from_state: State, goal_state: Goal, pathway_bias=None):

        if await goal_state.is_here(from_state):
            return goal_state

        pass

        # 1. generate action candidates
        # 2. evaluate action candidates
        # 3. s = s + a


    async def project_state(self, state: State):
        # return goal of the below layer
        pass