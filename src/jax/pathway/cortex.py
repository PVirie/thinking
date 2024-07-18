from humn.interfaces.pathway import cortex
from ..algebric import *
from typing import Tuple
import os

class Model(cortex.Cortex_Pathway):
    
    def __init__(self, model, step_discount_factor=0.9):
        super().__init__(step_discount_factor=step_discount_factor)
        self.model = model


    @staticmethod
    def load(path):
        pass

    def save(self, path):
        pass


    def refresh(self):
        pass



    def infer_sub_action(self, from_state: State, expect_action: Action) -> Tuple[Action, float]:
        goal_state = from_state + expect_action
        next_action_data, score = self.model.infer(from_state.data, goal_state.data)
        return Action(next_action_data), score


    def compute_masked_distance(self, indices, start, end):
        # return pow(self.step_discount_factor, < the distance from every item to the nearest future pivot >)
        pass


    def incrementally_learn(self, path: State_Sequence, pivots_indices: Index_Sequence):
        #  distances = self.compute_masked_distance(pivots_indices, 0, len(path))
        # learn to predict the next state and its probability from the current state given goal
        #  self.model.fit(path, path, distances)
        pass

