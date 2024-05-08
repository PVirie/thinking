from . import base
from .. import *


class Cortex_Pathway(base.Pathway):

    def __init__(self, model, step_discount_factor=0.9):
        self.model = model
        self.step_discount_factor = step_discount_factor

    
    def compute_masked_distance(self, indices, start, end):
        # return pow(self.step_discount_factor, < the distance from every item to the nearest future pivot >)
        pass


    def incrementally_learn(self, path: State_Sequence, pivots_indices: Index_Sequence):
        #  distances = self.compute_masked_distance(pivots_indices, 0, len(path))
        # learn to predict the next state and its probability from the current state given goal
        #  self.model.fit(path, path, distances)
        pass

