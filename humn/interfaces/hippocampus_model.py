from . import algebraic, trainer


class Model:

    def incrementally_learn(self, path: algebraic.State_Sequence) -> trainer.Trainer:
        raise NotImplementedError("Not implemented")
    

    def augmented_all(self) -> algebraic.Augmented_State_Squence:
        # return pathway encoding of all the states in the context
        raise NotImplementedError("Not implemented")


    def augment(self, path: algebraic.State_Sequence, pivot_indices: algebraic.Pointer_Sequence) -> algebraic.Augmented_State_Squence:
        # return pathway encoding of all the states in the context
        raise NotImplementedError("Not implemented")


    def append(self, state: algebraic.State) -> algebraic.State:
        # return the refined version of the state
        raise NotImplementedError("Not implemented")


    def refresh(self):
        raise NotImplementedError("Not implemented")


