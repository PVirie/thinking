from . import algebraic
from typing import Tuple, Union


class Model:


    def incrementally_learn(self, path: algebraic.Batch_State_Sequence) -> float:
        raise NotImplementedError("Not implemented")


    def abstract_path(self, path: algebraic.Batch_State_Sequence) -> Tuple[algebraic.Batch_Pointer_Sequence, algebraic.Batch_State_Sequence]:
        # return clusters of states under the same pivots
        # do not forget to include end
        raise NotImplementedError("Not implemented")


    def abstract(self, from_encoding_sequence: algebraic.Augmented_State_Squence, action: algebraic.Action) -> Tuple[algebraic.State, algebraic.Action]:
        # return next layer state and action
        raise NotImplementedError("Not implemented")


    def specify(self, nl_start: algebraic.State, nl_action: Union[algebraic.Action, None] = None, start: Union[algebraic.State, None] = None) -> Union[algebraic.Action, algebraic.State]:
        # inverse of abstract
        # return action if nl_action is not None, else return state
        raise NotImplementedError("Not implemented")
