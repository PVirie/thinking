from . import algebraic


class Model:

    
    def incrementally_learn(self, path_encoding_sequence: algebraic.Batch_Augmented_State_Squence, pivot_indices: algebraic.Batch_Pointer_Sequence, pivots: algebraic.Batch_State_Sequence) -> float:
        raise NotImplementedError("Not implemented")


    def infer_sub_action(self, from_encoding_sequence: algebraic.Augmented_State_Squence, expect_action: algebraic.Action) -> algebraic.Action:
        raise NotImplementedError("Not implemented")