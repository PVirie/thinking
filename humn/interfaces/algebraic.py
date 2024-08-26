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


class Batch_Pointer_Sequence:
    pass


class Batch_State_Sequence:

    def __len__(self):
        raise NotImplementedError("Not implemented")

    def sample_skip(self, n) -> Tuple[Batch_Pointer_Sequence, Batch_State_Sequence]:
        # if n is math.inf, return first and last indices
        raise NotImplementedError("Not implemented")


class Augmented_State_Squence:
    pass


class Batch_Augmented_State_Squence:
    pass
