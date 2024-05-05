from ..interface import State_Sequence


class Pathway:

    async def incrementally_learn(self, path: State_Sequence):
        pass
    

    async def infer_sub_action(self, from_state, expect_action):
        pass