from .interfaces import *
from typing import List, Tuple

class Layer:
    def __init__(self, cortex, hippocampus, use_entropy=True):
        self.heuristics = cortex
        self.hippocampus = hippocampus
        self.use_entropy = use_entropy


    def refresh(self):
        self.heuristics.refresh()
        self.hippocampus.refresh()


    def incrementally_learn_and_sample_pivots(self, path: State_Sequence):
        if len(path) < 2:
            return

        if self.use_entropy:
            indices, pivots = self.hippocampus.sample_local_entropy(path)
        else:
            indices, pivots = path.sample_skip(2, include_last = True)

        self.hippocampus.incrementally_learn(path, indices, pivots)
        self.heuristics.incrementally_learn(path, indices, pivots)

        return pivots


    def infer_sub_action(self, from_state: State, expect_action: Action, pathway_preference=None) -> Action:
        if from_state + expect_action == from_state:
            return expect_action

        heuristic_action, heuristic_score = self.heuristics.infer_sub_action(from_state, expect_action)
        hippocampus_action, hippocampus_score = self.hippocampus.infer_sub_action(from_state, expect_action)

        if pathway_preference is None:
            return heuristic_action if heuristic_score > hippocampus_score else hippocampus_action
        elif pathway_preference == "heuristics":
            return heuristic_action
        elif pathway_preference == "hippocampus":
            return hippocampus_action

