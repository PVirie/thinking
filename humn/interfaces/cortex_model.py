from . import algebraic


class Model:

    
    def incrementally_learn(self, path_encoding_sequence: algebraic.Augmented_State_Squence, pivot_indices: algebraic.Pointer_Sequence, pivots: algebraic.State_Sequence) -> float:
        return 0


    def infer_sub_action(self, from_encoding_sequence: algebraic.Augmented_State_Squence, expect_action: algebraic.Action) -> algebraic.Action:
        pass