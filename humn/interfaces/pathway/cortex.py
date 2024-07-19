from . import base
from .. import *


class Cortex_Pathway(base.Pathway):

    def __init__(self, step_discount_factor=0.9):
        self.step_discount_factor = step_discount_factor

    
    def incrementally_learn(self, path: State_Sequence, pivot_indices: Index_Sequence, pivots: State_Sequence):
        #  distances = pow(self.step_discount_factor, < the distance from every item to the nearest future pivot >)
        # learn to predict the next state and its probability from the current state given goal
        #  self.model.fit(path, path, distances)
        pass

