from __future__ import annotations
from typing import Tuple


class Action:
    def zero_length(self):
        pass


class State:
    pass


class Pointer_Sequence:
    pass


class State_Sequence:
    def sample_skip(self, n, include_last=False) -> Tuple[Pointer_Sequence, State_Sequence]:
        pass


class Augmented_State_Squence:
    pass

