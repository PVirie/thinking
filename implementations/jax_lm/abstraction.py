from humn import abstraction_model, trainer
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import os
import json

try:
    from .algebraic import *
except:
    from algebraic import *


def process_chunk(embedding_chunks, step_size=4):
    length = embedding_chunks.shape[0]
    # split chunks into sub_chunk of step_size
    pivot_size = math.ceil(length/step_size)
    data = jnp.pad(embedding_chunks, ((0, pivot_size * step_size - length), (0, 0)), mode='constant', constant_values=0)
    data = jnp.reshape(data, [pivot_size, step_size, -1])
    # average
    sizes = jnp.ones([pivot_size, 1], jnp.float32) * step_size
    sizes = sizes.at[-1].set(length - (pivot_size - 1) * step_size)
    abstract_chunks = jnp.sum(data, axis=1, keepdims=False) / sizes
    abstract_indices_chunks = [i for i in range(step_size, length, step_size)]
    return abstract_chunks, abstract_indices_chunks
    

class Model(abstraction_model.Model):

    def __init__(self, skip_steps):
        self.skip_steps = skip_steps

    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = Model(skip_steps=metadata["skip_steps"])
        return model                                               
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "skip_steps": self.skip_steps,
            }, f)


    def abstract_path(self, path: Embedding_Sequence) -> Tuple[Pointer_Sequence, Embedding_Sequence]:
        abstract_chunks, indices = process_chunk(path.data, step_size=self.skip_steps)
        return Pointer_Sequence(indices), Embedding_Sequence(abstract_chunks)


    def abstract(self, from_sequence: Augmented_Embedding_Squence, action: Text_Embedding) -> Tuple[Text_Embedding, Text_Embedding]:
        return Text_Embedding(from_sequence.data[-1, 0, :]), action


    def specify(self, nl_start: Text_Embedding, nl_action: Union[Text_Embedding, None] = None, start: Union[Text_Embedding, None] = None) -> Text_Embedding:
        return nl_start



if __name__ == "__main__":

    pivots, indices = process_chunk(jnp.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], jnp.float32), 3)

    print(indices)
    print(pivots)

