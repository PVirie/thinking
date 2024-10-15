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

            t = c.incrementally_learn(h.augment(path), pivot_indices, pivots)
            trainers.append(t)

        return trainers
    

    def __sub_action_recursion(self, i, state, action):
        if action.zero_length():
            return action
        
        cortex = self.cortices[i]
        hippocampus = self.hippocampi[i]
        abstractor = self.abstractors[i] if i < len(self.abstractors) else None

        hippocampus.append(state)

        if i < self.depth - 1:
            if abstractor is not None:
                nl_from_state, nl_action = abstractor.abstract(hippocampus.augmented_all(), action)
            else:
                nl_from_state, nl_action = state, action
            nl_sub_action = self.__sub_action_recursion(i + 1, nl_from_state, nl_action)
            if not nl_sub_action.zero_length():
                # if the next step is not the goal, specify the goal
                if abstractor is not None:
                    action = abstractor.specify(nl_from_state, nl_sub_action, state)
                else:
                    action = nl_sub_action

        return cortex.infer_sub_action(hippocampus.augmented_all(), action)


    def react(self, from_state: algebraic.State, top_action: algebraic.Action) -> algebraic.Action:
        return self.__sub_action_recursion(0, from_state, top_action)


    def __generate_steps(self, i, state, target_state):

        cortex = self.cortices[i]
        hippocampus = self.hippocampi[i]

        full_state = hippocampus.augmented_all()
        for step in range(self.max_sub_steps):
            target_action = target_state - full_state
            if target_action.zero_length():
                return
            sub_action = cortex.infer_sub_action(full_state, target_action)
            new_state = state + sub_action
            refined_state = hippocampus.append(new_state)
            full_state = hippocampus.augmented_all()
            yield refined_state, full_state

        raise MaxSubStepReached(f"Max sub step of {self.max_sub_steps} reached at layer {i}")

        
    def __think_recursion(self, i, state, action):
        if action.zero_length():
            return
    
        cortex = self.cortices[i]
        hippocampus = self.hippocampi[i]
        abstractor = self.abstractors[i] if i < len(self.abstractors) else None
        
        hippocampus.append(state)
        goal_state = state + action
        
        if i < self.depth - 1:

            if abstractor is not None:
                nl_from_state, nl_action = abstractor.abstract(hippocampus.augmented_all(), action)
            else:
                nl_from_state, nl_action = state, action
            goal_generator = self.__think_recursion(i + 1, nl_from_state, nl_action)
            
            while True:
                try:
                    nl_target_state = next(goal_generator)
                except StopIteration:
                    break

                if abstractor is not None:
                    target_state = abstractor.specify(nl_target_state)
                else:
                    target_state = nl_target_state
 
                for state, full_state in self.__generate_steps(i, state, target_state):
                    yield state
                    # goal_action = goal_state - full_state
                    # if goal_action.zero_length():
                    #     return
        
                if self.reset_hippocampus_on_target_changed:
                    hippocampus.refresh()
                    hippocampus.append(state)

        for state, full_state in self.__generate_steps(i, state, goal_state):
            yield state


    def think(self, from_state: algebraic.State, top_action: algebraic.Action) -> Generator[algebraic.State, None, None]:
        return self.__think_recursion(0, from_state, top_action)

