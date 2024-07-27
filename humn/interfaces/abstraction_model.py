from . import algebraic
from typing import Tuple


class Model:


    def incrementally_learn(self, path: algebraic.State_Sequence) -> float:
        return 0


    def abstract_path(self, path: algebraic.State_Sequence) -> Tuple[algebraic.Pointer_Sequence, algebraic.State_Sequence]:
        # return clusters of states under the same pivots
        # do not forget to include end
        pass


    def abstract(self, from_sequence: algebraic.State_Sequence, action: algebraic.Action) -> Tuple[algebraic.State, algebraic.Action]:
        # return next layer state and action
        pass


    def specify(self, start: algebraic.State, nl_start: algebraic.State, nl_action: algebraic.Action) -> algebraic.Action:
        # inverse of abstract
        # return action 
        pass
