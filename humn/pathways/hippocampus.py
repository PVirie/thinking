from . import base
from ..interface import State_Sequence, State, Action, Tensor, State_Matrix

class Hippocampus_Pathway(base.Pathway):

    def __init__(self, rows, cols):
        self.p_rel = Tensor(rows, cols)
        self.records = State_Matrix(rows, cols)

    def incrementally_learn(self, path: State_Sequence):
        
        self.records.roll(1, 0)
        self.records[-1, :] = path

        self.p_rel.roll([1, 0])
        # check whether this is relative jump or not


    def infer_sub_action(self, from_state: State, expect_action: Action):        
        goal_state = from_state + expect_action
