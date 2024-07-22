from . import algebraic
from typing import Tuple


class Model:

    
    def __call__(self, path: algebraic.State_Sequence) -> Tuple[algebraic.Index_Sequence, algebraic.State_Sequence]:
        pass


    def incrementally_learn(self, path: algebraic.State_Sequence):
        pass