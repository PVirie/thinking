from .interfaces import trainer, algebraic, cortex_model, hippocampus_model, abstraction_model
from typing import List, Tuple, Union, Generator, Any


# max sub step reach exception
class MaxSubStepReached(Exception):
    pass

class HUMN:
    def __init__(self, cortex_models: List[cortex_model.Model], hippocampus_models: List[hippocampus_model.Model], abstraction_models: List[abstraction_model.Model] = [], name="HUMN model"):
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
                    action = abstractor.specify(nl_from_state, nl_sub_action, abstractor)
                else:
                    action = nl_sub_action

        return cortex.infer_sub_action(hippocampus.augmented_all(), action)


    def infer_sub_action(self, from_state: algebraic.State, top_action: algebraic.Action) -> algebraic.Action:
        return self.__sub_action_recursion(0, from_state, top_action)


    def __generate_steps(self, i, state, target_state, max_sub_steps):

        cortex = self.cortices[i]
        hippocampus = self.hippocampi[i]

        for step in range(max_sub_steps):
            if state == target_state:
                return
            sub_action = cortex.infer_sub_action(hippocampus.augmented_all(), target_state - state)
            state = state + sub_action
            hippocampus.append(state)
            yield state

        raise MaxSubStepReached(f"Max sub step of {max_sub_steps} reached at layer {i}")

        
    def __think_recursion(self, i, state, action, max_sub_steps):
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
            goal_generator = self.__think_recursion(i + 1, nl_from_state, nl_action, max_sub_steps)
            
            while True:
                try:
                    nl_target_state = next(goal_generator)
                except StopIteration:
                    break

                if abstractor is not None:
                    target_state = abstractor.specify(nl_target_state)
                else:
                    target_state = nl_target_state

                yield from self.__generate_steps(i, state, target_state, max_sub_steps)
                state = target_state
        
        yield from self.__generate_steps(i, state, goal_state, max_sub_steps)


    def think(self, from_state: algebraic.State, top_action: algebraic.Action, max_sub_steps=128) -> Generator[algebraic.State, None, None]:
        return self.__think_recursion(0, from_state, top_action, max_sub_steps)

