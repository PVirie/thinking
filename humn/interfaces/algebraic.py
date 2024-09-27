from __future__ import annotations
from typing import Tuple


class Action:
    def zero_length(self):
        raise NotImplementedError("Not implemented")

    # interface for think method
    def __add__(self, s): 
        return s + self
    
class State:

    # interface for think method
    def __add__(self, a):
        raise NotImplementedError("Not implemented")


    # interface for think method
    def __sub__(self, s):
        raise NotImplementedError("Not implemented")


    # interface for think method
    def __eq__(self, s):
        raise NotImplementedError("Not implemented")


class Pointer_Sequence:
    pass


class State_Sequence:
    pass


class Augmented_State_Squence:
    pass

