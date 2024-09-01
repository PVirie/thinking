from . import algebraic, trainer
from typing import Tuple, Union


class Model:


    def incrementally_learn(self, path: algebraic.State_Sequence) -> trainer.Trainer:
        raise NotImplementedError("Not implemented")


    def abstract_path(self, path: algebraic.State_Sequence) -> Tuple[algebraic.Pointer_Sequence, algebraic.State_Sequence]:
        # return clusters of states under the same pivots
        # do not forget to include end
        raise NotImplementedError("Not implemented")


    def abstract(self, from_sequence: algebraic.Augmented_State_Squence, action: algebraic.Action) -> Tuple[algebraic.State, algebraic.Action]:
        # return next layer state and action
        raise NotImplementedError("Not implemented")


    def specify(self, nl_start: algebraic.State, nl_action: Union[algebraic.Action, None] = None, start: Union[algebraic.State, None] = None) -> Union[algebraic.Action, algebraic.State]:
        # inverse of abstract
        # return action if nl_action is not None, else return state
        raise NotImplementedError("Not implemented")
