import logging
from .interfaces import algebraic, cortex_model, hippocampus_model, abstraction_model
from .layer import Layer
from typing import List, Tuple



class HUMN:
    def __init__(self, layers: List[Layer]):
        self.layers = layers


    def refresh(self):
        for layer in self.layers:
            layer.refresh()


    def observe(self, path: algebraic.State_Sequence):
        current_layer_path = path        
        for layer in self.layers:
            current_layer_path = layer.incrementally_learn_and_sample_pivots(current_layer_path)


    def think(self, from_state: algebraic.State, goal_state: algebraic.State):
        if len(self.layers) == 0:
            logging.error("No layers in HUMN, please initialize it.")
            return None
        
        if from_state == goal_state:
            return goal_state

        action = goal_state - from_state
        for layer in self.layers:
            action = layer.abstract(action)

        for layer in reversed(self.layers):
            abstract_action = layer.infer_sub_action(from_state, action)
            action = layer.specify(abstract_action)


        return from_state + action

