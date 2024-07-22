from . import algebraic


class Model:

    
    def __call__(self) -> algebraic.Augmented_State_Squence:
        # return pathway encoding of all the states in the context
        pass


    def append(self, state: algebraic.State):
        pass


    def extend(self, path: algebraic.State_Sequence):
        pass


    def refresh(self):
        pass


