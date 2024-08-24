from . import algebraic


class Model:

    
    def augmented_all(self) -> algebraic.Augmented_State_Squence:
        # return pathway encoding of all the states in the context
        raise NotImplementedError("Not implemented")


    def all(self) -> algebraic.State_Sequence:
        # return pathway encoding of all the states in the context
        raise NotImplementedError("Not implemented")


    def append(self, state: algebraic.State):
        raise NotImplementedError("Not implemented")


    def extend(self, path: algebraic.State_Sequence):
        raise NotImplementedError("Not implemented")


    def refresh(self):
        raise NotImplementedError("Not implemented")


