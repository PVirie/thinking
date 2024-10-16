from humn import hippocampus_model
from typing import Tuple
import jax.numpy as jnp

import os
import json

from .algebraic import *


class Model(hippocampus_model.Model):

    def __init__(self, dims):
        self.dims = dims
        self.data = jnp.zeros(dims, dtype=jnp.float32)


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = Model(metadata["dims"])
        return model
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "dims": self.dims
            }, f)



    def augmented_all(self) -> Cart_State:
        return Cart_State(
            self.data
        )


    def augment(self, path: State_Action_Sequence, pivot_sequence: Pointer_Sequence) -> State_Action_Sequence:
        return path


    def append(self, state: Cart_State):
        self.data = state.data
        return state


    def refresh(self):
        self.data = self.data.at[:].set(0.0)

