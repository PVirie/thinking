from . import algebraic


class Model:

    
    def incrementally_learn(self, path_encoding_sequence: algebraic.Augmented_State_Squence, pivot_indices: algebraic.Index_Sequence, pivots: algebraic.State_Sequence):
        #  distances = pow(self.step_discount_factor, < the distance from every item to the nearest future pivot >)
        # learn to predict the next state and its probability from the current state given goal
        #  self.model.fit(path, path, distances)
        pass


    def infer_sub_action(self, from_encoding_sequence: algebraic.Augmented_State_Squence, expect_action: algebraic.Action) -> algebraic.Action:
        pass