from humn.interfaces.pathway import hippocampus
from ..algebric import *
from typing import Tuple

class Model(hippocampus.Hippocampus_Pathway):
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def load(path):
        pass

    def save(self, path):
        pass


    def refresh(self):
        pass


    def infer_sub_action(self, from_state: State, expect_action: Action) -> Tuple[Action, float]:
        # goal_state = from_state + expect_action
        # next_action, score = self.model.infer(from_state, goal_state)
        # return next_action, score
        pass


    def compute_entropy_local_minimum_indices(self, path: State_Sequence) -> Index_Sequence:
        pass
    

    def incrementally_learn(self, path: State_Sequence, pivots_indices: Index_Sequence):
        # compute gap distance between each consecutive states
        # if 99.7% of the distance less than 1 unit is within 3 std, then it is a relative jump
        pass


