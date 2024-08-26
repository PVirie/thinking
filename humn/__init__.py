from .interfaces import algebraic, cortex_model, hippocampus_model, abstraction_model
from .layer import Layer
from typing import List, Tuple, Union, Generator



class HUMN:
    def __init__(self, layers: List[Layer], abstraction_models: List[abstraction_model.Model] = [], name="HUMN model"):
        if layers == None or len(layers) == 0:
            raise ValueError("At least one layer is required")
        
        # fill abstraction model with none to have size len(layers) - 1
        while len(abstraction_models) < len(layers) - 1:
            abstraction_models.append(None)

        for l, l_, ab in zip(layers[:-1], layers[1:], abstraction_models):
            l.set_next_layer(l_, ab)
        
        self.root = layers[0]
        self.name = name
        

    def refresh(self):
        self.root.refresh()


    def observe(self, path: algebraic.Batch_State_Sequence):
        self.root.incrementally_learn(path)


    def infer_sub_action(self, from_state: algebraic.State, action: algebraic.Action) -> algebraic.Action:
        return self.root.infer_sub_action(from_state, action)


    def think(self, from_state: algebraic.State, action: algebraic.Action) -> Generator[algebraic.State, None, None]:
        return self.root.think(from_state, action)

