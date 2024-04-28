from .interface import *
from typing import List

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


    async def incrementally_learn(self, path: List[From_State], pivots_indices):
        if len(path) < 2:
            return

        await self.hippocampus.incrementally_learn(path)
        await self.proxy.incrementally_learn(path)

        # await self.heuristics.incrementally_learn(path, pivots_indices)
        hippocampus_distances = await self.hippocampus.distance(path, [path[i] for i in pivots_indices])
        await self.heuristics.incrementally_learn_2(path, pivots_indices, hippocampus_distances)



    async def next_step(self, from_state: From_State, goal_state: To_State, pathway_bias=None):

        if await goal_state.is_here(from_state):
            return goal_state
        

        candidates, props = await self.proxy.get_candidates(from_state)

        cortex_rep, cortex_prop = await self.heuristics.consolidate(from_state, candidates, props, goal_state)
        hippocampus_rep, hippocampus_prop = await self.hippocampus.infer(from_state, goal_state)

        # sup_info = [(await self.hippocampus.enhance(cortex_rep), cortex_prop), (await self.hippocampus.enhance(hippocampus_rep), hippocampus_prop)]

        if pathway_bias is None:
            return hippocampus_rep if hippocampus_prop > cortex_prop else cortex_rep
        elif pathway_bias == "hippocampus":
            return hippocampus_rep
        if pathway_bias == "cortex":
            return cortex_rep

