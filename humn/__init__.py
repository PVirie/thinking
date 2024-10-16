from .interfaces import trainer, algebraic, cortex_model, hippocampus_model, abstraction_model
from typing import List, Tuple, Union, Generator, Any


# max sub step reach exception
class MaxSubStepReached(Exception):
    pass

class HUMN:
    def __init__(self, 
                 cortex_models: List[cortex_model.Model], 
                 hippocampus_models: List[hippocampus_model.Model], 
                 abstraction_models: List[abstraction_model.Model] = [], 
                 name="HUMN model", reset_hippocampus_on_target_changed=True, max_sub_steps=16):
        if len(cortex_models) == 0 or len(hippocampus_models) == 0:
            raise ValueError("At least one layer is required")
        
        self.depth = len(cortex_models)
        
        # fill abstraction model with none to have size len(layers) - 1
        while len(abstraction_models) < self.depth - 1:
            abstraction_models.append(None)

        self.cortices = cortex_models
        self.hippocampi = hippocampus_models
        self.abstractors = abstraction_models

        self.name = name
        self.reset_hippocampus_on_target_changed = reset_hippocampus_on_target_changed
        self.max_sub_steps = max_sub_steps
        

    def refresh(self):
        for h in self.hippocampi:
            h.refresh()


    def observe(self, path_tuples: List[Tuple[algebraic.State_Sequence, algebraic.Pointer_Sequence, algebraic.State_Sequence]]) -> List[trainer.Trainer]:
        # not learning abstraction here, must be done externally
        if len(path_tuples) < self.depth:
            raise ValueError("Not enough layer data to learn")
        
        # learn, can be parallelized
        trainers = []
        for i, (path, pivot_indices, pivots) in enumerate(path_tuples[:self.depth]):
            c = self.cortices[i]
            h = self.hippocampi[i]

            t = c.incrementally_learn(h.augment(path, pivot_indices), pivot_indices, pivots)
            trainers.append(t)

        return trainers
    

    def react(self, from_states: Union[algebraic.State, List[algebraic.State]], top_action: algebraic.Action) -> algebraic.Action:
        if not isinstance(from_states, List):
            # duplicate to the number of layers
            base = from_states
            from_states = [base]
            for i in range(0, self.depth - 1):
                from_states.append(base if self.abstractors[i] is None else self.abstractors[i].abstract_start(base))

        for i in range(self.depth - 1, -1, -1):
            self.hippocampi[i].append(from_states[i])
            if i < self.depth - 1 and self.abstractors[i] is not None:
                top_action = self.abstractors[i].specify(action)
            if not top_action.zero_length():
                action = self.cortices[i].infer_sub_action(self.hippocampi[i].augmented_all(), top_action)
            else:
                action = top_action
        return action


    def __generate_steps(self, i, state, target_state):

        cortex = self.cortices[i]
        hippocampus = self.hippocampi[i]

        if self.reset_hippocampus_on_target_changed:
            hippocampus.refresh()

        output_states = []
        for _ in range(self.max_sub_steps):
            hippocampus.append(state)
            full_state = hippocampus.augmented_all()
            target_action = target_state - full_state
            if target_action.zero_length():
                break
            sub_action = cortex.infer_sub_action(full_state, target_action)
            state = state + sub_action
            output_states.append(state)
            if sub_action.zero_length():
                break
        else:
            raise MaxSubStepReached(f"Max sub step of {self.max_sub_steps} reached at layer {i}")
        
        return output_states, state

        
    def think(self, from_states: Union[algebraic.State, List[algebraic.State]], top_action: algebraic.Action) -> Generator[algebraic.State, None, None]:
        if not isinstance(from_states, List):
            # duplicate to the number of layers
            base = from_states
            from_states = [base]
            for i in range(0, self.depth - 1):
                from_states.append(base if self.abstractors[i] is None else self.abstractors[i].abstract_start(base))

        top_states = [from_states[-1] + top_action]
        for i in range(self.depth - 1, -1, -1):
            if i < self.depth - 1 and self.abstractors[i] is not None:
                abstractor = self.abstractors[i]
                top_states = [abstractor.specify(s) for s in top_states] 
            current_state = from_states[i]
            states = []
            for top_state in top_states:
                state_batches, current_state = self.__generate_steps(i, current_state, top_state)
                states.extend(state_batches)
            top_states = states
        return states

