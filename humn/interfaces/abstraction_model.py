from . import algebraic
from typing import Tuple, Union


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


    def specify(self, nl_start: algebraic.State, nl_action: Union[algebraic.Action, None] = None, start: Union[algebraic.State, None] = None) -> Union[algebraic.Action, algebraic.State]:
        # inverse of abstract
        # return action if nl_action is not None, else return state
        pass
