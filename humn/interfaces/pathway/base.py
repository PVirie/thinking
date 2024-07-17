from .. import *
from typing import Tuple


class Pathway:

    def refresh(self):
        pass

    def incrementally_learn(self, path: State_Sequence, pivots_indices: Index_Sequence):
        pass
    

    def infer_sub_action(self, from_state: State, expect_action: Action) -> Tuple[Action, float]:
        # goal_state = from_state + expect_action
        # next_action, score = self.model.infer(from_state, goal_state)
        # return next_action, score
        pass