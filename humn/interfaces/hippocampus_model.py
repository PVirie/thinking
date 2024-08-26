from . import algebraic


class Model:

    
    def batch_augmented_all(self, batch_sequence: algebraic.Batch_State_Sequence) -> algebraic.Batch_Augmented_State_Squence:
        # return pathway encoding of all the states in the context
        raise NotImplementedError("Not implemented")


    def augmented_all(self) -> algebraic.Augmented_State_Squence:
        # return pathway encoding of all the states in the context
        raise NotImplementedError("Not implemented")


    def append(self, state: algebraic.State):
        raise NotImplementedError("Not implemented")


    def refresh(self):
        raise NotImplementedError("Not implemented")


