from . import algebraic
from typing import Tuple


class Model:

    

    def incrementally_learn(self, path: algebraic.State_Sequence) -> float:
        return 0


    def abstract(self, path: algebraic.State_Sequence) -> Tuple[algebraic.Index_Sequence, algebraic.State_Sequence]:
        pass


    def abstract(self, action: algebraic.Action) -> algebraic.Action:
        pass


    # inverse of abstract
    def specify(self, action: algebraic.Action) -> algebraic.Action:
        pass
