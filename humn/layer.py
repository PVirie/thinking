from .interfaces import algebraic
from .interfaces import cortex_model, hippocampus_model, abstraction_model
from typing import List, Tuple, Union

class Layer:
    def __init__(self, 
                 cortex_model: cortex_model.Model, 
                 hippocampus_model: hippocampus_model.Model, 
                 abstraction_model: Union[abstraction_model.Model, None] = None):
        self.cortex = cortex_model
        self.hippocampus_model = hippocampus_model
        self.abstraction_model = abstraction_model


    def refresh(self):
        self.hippocampus_model.refresh()


    def incrementally_learn_and_sample_pivots(self, path: algebraic.State_Sequence) -> algebraic.State_Sequence:
        if len(path) < 2:
            return

        if self.abstraction_model is not None:
            indices, pivots = self.abstraction_model(path)
            self.abstraction_model.incrementally_learn(path)
        else:
            indices, pivots = path.sample_skip(2, include_last = True)

        self.refresh()
        self.hippocampus_model.extend(path)
        self.cortex.incrementally_learn(self.hippocampus_model(), indices)

        return pivots


    def infer_sub_action(self, from_state: algebraic.State, expect_action: algebraic.Action) -> algebraic.Action:
        if from_state + expect_action == from_state:
            # reach goal state
            return expect_action

        self.hippocampus_model.append(from_state)
        return self.cortex.infer_sub_action(self.hippocampus_model(), expect_action)

