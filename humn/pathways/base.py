from ..interface import State_Sequence, State, Action


class Pathway:

    def incrementally_learn(self, path: State_Sequence):
        pass
    

    def infer_sub_action(self, from_state: State, expect_action: Action):
        pass