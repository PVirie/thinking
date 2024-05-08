import logging
from .interfaces import *
from .layer import Layer
from typing import List, Tuple



def compute_pivot_indices(layer, path: State_Sequence, use_entropy=False) -> Index_Sequence:
    if use_entropy:
        all_pvs = layer.compute_entropy_local_minima(path)
    else:
        # skip by 1
        all_pvs = Index_Sequence()
        for i in range(1, len(path), 2):
            all_pvs.append(i)
    
    # always have the last
    all_pvs.append(len(path))
    return all_pvs


class HUMN:
    def __init__(self, layers):
        self.layers = layers


    def observe(self, path: State_Sequence):
        current_layer_path = path        
        for layer in self.layers:
            pivots_indices = compute_pivot_indices(current_layer_path)
            layer.incrementally_learn(current_layer_path, pivots_indices)
            current_layer_path = current_layer_path.generate_subsequence(pivots_indices)


    def think(self, from_state: State, goal_state: State):
        if len(self.layers) == 0:
            logging.error("No layers in HUMN, please initialize it.")
            return None
        
        if from_state == goal_state:
            return goal_state

        action = goal_state - from_state
        for layer in reversed(self.layers):
            action = layer.infer_sub_action(from_state, action)

        return from_state + action


    def save(self, path):
        pass


    def load(self, path):
        pass


    def set_pathway_preference(self, pathway_name, preference):
        pass