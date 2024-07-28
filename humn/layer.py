from .interfaces import algebraic, cortex_model, hippocampus_model, abstraction_model
from typing import List, Tuple, Union

class Layer:
    def __init__(self, cortex_model: cortex_model.Model, hippocampus_model: hippocampus_model.Model):
        self.cortex_model = cortex_model
        self.hippocampus_model = hippocampus_model
        self.next_layer = None
        self.abstraction_model = None


    def refresh(self):
        self.hippocampus_model.refresh()
        if self.next_layer is not None:
            self.next_layer.refresh()


    def set_next_layer(self, next_layer: 'Layer', abstraction_model: Union[abstraction_model.Model, None] = None):
        self.next_layer = next_layer
        self.abstraction_model = abstraction_model
    

    def incrementally_learn(self, path: algebraic.State_Sequence):
        if len(path) < 2:
            return

        if self.abstraction_model is not None:
            clusters, pivots = self.abstraction_model.abstract_path(path)
            self.abstraction_model.incrementally_learn(path)
        else:
            clusters, pivots = path.sample_skip(2, include_last = True)

        self.refresh()
        self.hippocampus_model.extend(path)
        self.cortex_model.incrementally_learn(self.hippocampus_model.augmented_all(), clusters, pivots)

        if self.next_layer is not None:
            self.next_layer.incrementally_learn(pivots)


    def infer_sub_action(self, from_state: algebraic.State, action: algebraic.Action) -> algebraic.Action:
        if action.zero_length():
            return action
        
        self.hippocampus_model.append(from_state)

        if self.next_layer is not None:
            if self.abstraction_model is not None:
                nl_from_state, nl_action = self.abstraction_model.abstract(self.hippocampus_model.all(), action)
            else:
                nl_from_state, nl_action = from_state, action
            nl_sub_action = self.next_layer.infer_sub_action(nl_from_state, nl_action)
            if not nl_sub_action.zero_length():
                # if the next step is not the goal, specify the goal
                if self.abstraction_model is not None:
                    action = self.abstraction_model.specify(from_state, nl_from_state, nl_sub_action)
                else:
                    action = nl_sub_action

        return self.cortex_model.infer_sub_action(self.hippocampus_model.augmented_all(), action)


