from __future__ import annotations
from typing import Tuple

class Action:
    def __add__(self, s): 
        return s + self
    
    def zero_length(self):
        pass


class State:
    def __add__(self, a): 
        pass

    def __sub__(self, s):
        pass

    def __eq__(self, s):
        return (self - s).zero_length()



class Index_Sequence:
    pass


class State_Sequence:
    def sample_skip(self, n, include_last=False) -> Tuple[Index_Sequence, State_Sequence]:
        pass


class Augmented_State_Squence:
    pass

