from . import algebraic


class Model:

    
    def augmented_all(self) -> algebraic.Augmented_State_Squence:
        # return pathway encoding of all the states in the context
        pass


    def all(self) -> algebraic.State_Sequence:
        # return pathway encoding of all the states in the context
        pass


    def append(self, state: algebraic.State):
        pass


    def extend(self, path: algebraic.State_Sequence):
        pass


    def refresh(self):
        pass


