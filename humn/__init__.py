import logging
from .interface import *
from .layer import Layer
from typing import List, Tuple



async def compute_pivot_indices(layer, path, use_entropy=False):
    if use_entropy:
        entropies = await layer.compute_entropy(path)
        all_pvs = []
        for j in range(1, len(entropies) - 1):
            if entropies[j] > entropies[j + 1]:
                all_pvs.append(j)
        all_pvs.append(len(entropies) - 1)
    else:
        # skip by 1
        all_pvs = list(range(1, len(path), 2))
    return all_pvs


class HUMN:
    def __init__(self):
        self.layers = []


    async def observe(self, path: State_Sequence):
        current_layer_path = path        
        for layer in self.layers:
            pivots_indices = await compute_pivot_indices(current_layer_path)
            await layer.incrementally_learn(current_layer_path, pivots_indices)
            current_layer_path = [current_layer_path[i] for i in pivots_indices]


    async def think(self, from_state: State, goal_state: Goal):
        if len(self.layers) == 0:
            logging.error("No layers in HUMN, please initialize it.")
            return None
        
        if await goal_state.is_here(from_state):
            return goal_state

        last_layer_goal = goal_state
        for layer in reversed(self.layers):
            layer_next_state = await layer.infer_next_step(from_state, last_layer_goal)
            last_layer_goal = await layer.project_state(layer_next_state)

        return layer_next_state


    def save(self, path):
        pass


    def load(self, path):
        pass

    def set_pathway_preference(self, pathway_name, preference):
        pass