from . import base
from .. import *
from typing import Tuple

class Hippocampus_Pathway(base.Pathway):

    def __init__(self):
        pass


    def sample_local_entropy(self, path: State_Sequence) -> Tuple[Index_Sequence, State_Sequence]:
        pass
    

    def incrementally_learn(self, path: State_Sequence, pivot_indices: Index_Sequence, pivots: State_Sequence):
        # compute gap distance between each consecutive states
        # if 99.7% of the distance less than 1 unit is within 3 std, then it is a relative jump
        pass


