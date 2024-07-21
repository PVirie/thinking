from .. import *
from typing import Tuple


class Pathway:

    def __init__(self, step_discount_factor=0.9):
        self.step_discount_factor = step_discount_factor
    
    
    def refresh(self):
        pass


    def incrementally_learn(self, path: State_Sequence, pivot_indices: Index_Sequence, pivots: State_Sequence):
        pass
    

    def infer_sub_action(self, from_state: State, expect_action: Action) -> Tuple[Action, float]:
        # goal_state = from_state + expect_action
        # next_action, score = self.model.infer(from_state, goal_state)
        # return next_action, score
        pass