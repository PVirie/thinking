import logging
from .interface import *
from .layer import Layer
from typing import List



def compute_pivot_indices(entropies):
    all_pvs = []
    for j in range(1, len(entropies) - 1):
        if entropies[j] > entropies[j + 1]:
            all_pvs.append(j)
    all_pvs.append(len(entropies) - 1)
    return all_pvs


class HUMN:
    def __init__(self):
        self.layers = []


    async def observe(self, path: List[From_State]):
        current_layer_path = path        
        for layer in self.layers:
            entropies = await self.hippocampus.compute_entropy(current_layer_path)
            pivots_indices = compute_pivot_indices(entropies)

            await layer.incrementally_learn(current_layer_path, pivots_indices)

            current_layer_path = [current_layer_path[i] for i in pivots_indices]



    async def think(self, from_state: From_State, goal_state: To_State):
        if len(self.layers) == 0:
            logging.error("No layers in HUMN, please initialize it.")
            return None
        
        if await goal_state.is_here(from_state):
            return goal_state

        from_state_projections = []

        f = from_state
        last_layer_goal = goal_state
        for layer in self.layers:
            from_state_projections.append(f)

            # perform forward sampling for all variables
            # find the local maxima of the entropy
            # keep the values as the next layer projection
            # f = projection_up(f, backward=False)
            # last_layer_goal = projection_up(last_layer_goal, backward=True)
            pass

        last_layer_goal
        for layer, f in reversed(zip(self.layers, from_state_projections)):
            g = project_down(last_layer_goal)
            if g.is_here(f):
                last_layer_goal = g
            else:
                last_layer_goal = await layer.next_step(f, g)
            
        return last_layer_goal


    def save(self, path):
        pass


    def load(self, path):
        pass

    def set_pathway_preference(self, pathway_name, preference):
        pass