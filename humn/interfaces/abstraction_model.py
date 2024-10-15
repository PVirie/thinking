from . import algebraic, trainer
from typing import Tuple, Union


class Model:


    def incrementally_learn(self, path: algebraic.State_Sequence) -> trainer.Trainer:
        raise NotImplementedError("Not implemented")


    def abstract_path(self, path: algebraic.State_Sequence) -> Tuple[algebraic.Pointer_Sequence, algebraic.State_Sequence]:
        # return clusters of states under the same pivots
        raise NotImplementedError("Not implemented")


    def abstract_start(self, state: algebraic.State) -> algebraic.State:
        # return the start state of the path
        raise NotImplementedError("Not implemented")


    def specify(self, nl_action: algebraic.Action) -> algebraic.Action:
        # inverse of abstract
        # return action if nl_action is not None, else return state
        raise NotImplementedError("Not implemented")
