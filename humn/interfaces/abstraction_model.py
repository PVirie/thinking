from . import algebraic
from typing import Tuple


class Model:


    def incrementally_learn(self, path: algebraic.State_Sequence) -> float:
        return 0


    def abstract_path(self, path: algebraic.State_Sequence) -> Tuple[algebraic.Index_Sequence, algebraic.State_Sequence]:
        pass


    def abstract(self, start: algebraic.State, action: algebraic.Action) -> Tuple[algebraic.State, algebraic.Action]:
        pass


    # inverse of abstract
    def specify(self, start: algebraic.State, goal: algebraic.Action) -> algebraic.Action:
        pass
