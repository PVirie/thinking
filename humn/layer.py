from .interfaces import algebraic, cortex_model, hippocampus_model, abstraction_model
from typing import List, Tuple, Union, Generator
import math

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
    

    def incrementally_learn(self, path: algebraic.Batch_State_Sequence):
        if len(path) < 2:
            return

        if self.next_layer is not None:
            if self.abstraction_model is not None:
                clusters, pivots = self.abstraction_model.abstract_path(path)
                self.abstraction_model.incrementally_learn(path)
            else:
                clusters, pivots = path.sample_skip(2)
        else:
            clusters, pivots = path.sample_skip(math.inf)

        if self.next_layer is not None:
            self.next_layer.incrementally_learn(pivots)

        self.cortex_model.incrementally_learn(self.hippocampus_model.batch_augmented_all(path), clusters, pivots)



    def infer_sub_action(self, from_state: algebraic.State, action: algebraic.Action) -> algebraic.Action:
        if action.zero_length():
            return action
        
        self.hippocampus_model.append(from_state)

        if self.next_layer is not None:
            if self.abstraction_model is not None:
                nl_from_state, nl_action = self.abstraction_model.abstract(self.hippocampus_model.augmented_all(), action)
            else:
                nl_from_state, nl_action = from_state, action
            nl_sub_action = self.next_layer.infer_sub_action(nl_from_state, nl_action)
            if not nl_sub_action.zero_length():
                # if the next step is not the goal, specify the goal
                if self.abstraction_model is not None:
                    action = self.abstraction_model.specify(nl_from_state, nl_sub_action, from_state)
                else:
                    action = nl_sub_action

        return self.cortex_model.infer_sub_action(self.hippocampus_model.augmented_all(), action)



    def __generate_steps(self, state, target_state):
        while True:
            self.hippocampus_model.append(state)
            sub_action = self.cortex_model.infer_sub_action(self.hippocampus_model.augmented_all(), target_state - state)
            state = state + sub_action
            yield state
            if state == target_state:
                break
            

    def think(self, from_state: algebraic.State, action: algebraic.Action) -> Generator[algebraic.State, None, None]:
        if action.zero_length():
            return
    
        state = from_state
        goal_state = state + action
        

        if self.next_layer is not None:

            if self.abstraction_model is not None:
                nl_from_state, nl_action = self.abstraction_model.abstract(self.hippocampus_model.augmented_all(), action)
            else:
                nl_from_state, nl_action = from_state, action
            goal_generator = self.next_layer.think(nl_from_state, nl_action)
            
            state = from_state
            while True:
                try:
                    nl_target_state = next(goal_generator)
                except StopIteration:
                    break
                if self.abstraction_model is not None:
                    target_state = self.abstraction_model.specify(nl_target_state)
                else:
                    target_state = nl_target_state

                yield from self.__generate_steps(state, target_state)
                state = target_state
        
        yield from self.__generate_steps(state, goal_state)
