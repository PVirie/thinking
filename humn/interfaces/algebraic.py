from __future__ import annotations
from typing import Tuple


class Action:
    def zero_length(self):
        raise NotImplementedError("Not implemented")


class Pointer_Sequence:
    pass


class State_Sequence:
    pass


class Augmented_State_Squence:
    pass


class State:

    # interface for think method
    def __add__(self, a: Action) -> State:
        raise NotImplementedError("Not implemented")


    # interface for think method
    def __sub__(self, augmented_state_sequence: Augmented_State_Squence) -> Action:
        raise NotImplementedError("Not implemented")



