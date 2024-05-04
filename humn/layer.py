from .interface import *
from typing import List, Tuple

class Layer:
    def __init__(self, name, heuristics, hippocampus, proxy):
        self.name = name
        self.heuristics = heuristics
        self.hippocampus = hippocampus


    def save(self, weight_path):
        pass

    def load(self, weight_path):
        pass


    async def incrementally_learn(self, path: State_Sequence, pivots_indices):
        if len(path) < 2:
            return

        await self.hippocampus.incrementally_learn(path)

        # await self.heuristics.incrementally_learn(path, pivots_indices)

        hippocampus_distances = await self.hippocampus.distance(path, path.generate_subsequence(pivots_indices).unroll())
        await self.heuristics.incrementally_learn_2(path, pivots_indices, hippocampus_distances)


    async def infer_sub_action(self, from_state: State, expect_action: Action, pathway_bias=None):

        if from_state + expect_action == from_state:
            return expect_action

        heuristic_state, heuristic_score = await self.heuristics.infer_sub_action(from_state, expect_action)
        hippocampus_state, hippocampus_score = await self.hippocampus.infer_sub_action(from_state, expect_action)

        if pathway_bias is None:
            return heuristic_state if heuristic_score > hippocampus_score else hippocampus_state
        elif pathway_bias == "heuristics":
            return heuristic_state
        elif pathway_bias == "hippocampus":
            return hippocampus_state

