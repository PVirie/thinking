from .interfaces import *
from typing import List, Tuple

class Layer:
    def __init__(self, cortex, hippocampus):
        self.heuristics = cortex
        self.hippocampus = hippocampus


    def save(self, weight_path):
        pass

    def load(self, weight_path):
        pass

    
    def compute_entropy_local_minima(self, path: State_Sequence) -> Index_Sequence:
        return self.hippocampus.compute_entropy_local_minimum_indices(path)


    def incrementally_learn(self, path: State_Sequence, pivots_indices: Index_Sequence):
        if len(path) < 2:
            return

        self.hippocampus.incrementally_learn(path, pivots_indices)
        self.heuristics.incrementally_learn(path, pivots_indices)


    def infer_sub_action(self, from_state: State, expect_action: Action, pathway_bias=None) -> Action:

        if from_state + expect_action == from_state:
            return expect_action

        heuristic_action, heuristic_score = self.heuristics.infer_sub_action(from_state, expect_action)
        hippocampus_action, hippocampus_score = self.hippocampus.infer_sub_action(from_state, expect_action)

        if pathway_bias is None:
            return heuristic_action if heuristic_score > hippocampus_score else hippocampus_action
        elif pathway_bias == "heuristics":
            return heuristic_action
        elif pathway_bias == "hippocampus":
            return hippocampus_action

