from . import base
from .. import *

class Hippocampus_Pathway(base.Pathway):

    def __init__(self):
        pass


    def compute_entropy_local_minimum_indices(self, path: State_Sequence) -> Index_Sequence:
        pass
    

    def incrementally_learn(self, path: State_Sequence, pivots_indices: Index_Sequence):
        # compute gap distance between each consecutive states
        # if 99.7% of the distance less than 1 unit is within 3 std, then it is a relative jump
        pass


