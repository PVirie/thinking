import jax.numpy as jnp
import jax.random
from jax import device_put
from typing import List
import humn
import os
import math


    
class Pointer_Sequence(humn.algebraic.Pointer_Sequence):
    def __init__(self, indices = None):
        if indices is None:
            self.data = None
        else:
            self.data = jnp.array(indices, dtype=jnp.int32)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, s):
        self.data = self.data.at[i].set(s)
        return self

    def append(self, s):
        if self.data is None:
            self.data = jnp.array([s], dtype=jnp.int32)
        else:
            self.data = jnp.concatenate([self.data, jnp.array([s], dtype=jnp.int32)], axis=0)
        return self

    def __len__(self):
        return self.data.shape[0]
    

class Embedding_Sequence(humn.algebraic.State_Sequence):
    def __init__(self, states = None):
        if states is None:
            self.data = None
        elif isinstance(states, List):
            self.data = jnp.array([s.data for s in states])
        elif isinstance(states, jnp.ndarray):
            self.data = states
        else:
            self.data = device_put(jnp.array(states, jnp.float32))

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, i):
        if isinstance(i, slice):
            return Embedding_Sequence(self.data[i])
        elif isinstance(i, Pointer_Sequence):
            return Embedding_Sequence(self.data[i.data])
        else:
            return Text_Embedding(self.data[i, :])


    def __setitem__(self, i, s):
        self.data = self.data.at[i].set(s.data)
        return self


    def prepend(self, s):
        # s is of Text_Embedding type
        if self.data is None:
            self.data = jnp.reshape(s.data, [1, -1])
        else:
            self.data = jnp.concatenate([jnp.reshape(s.data, [1, -1]), self.data], axis=0)
        return self


    def append(self, s):
        # s is of Text_Embedding type
        if self.data is None:
            self.data = jnp.reshape(s.data, [1, -1])
        else:
            self.data = jnp.concatenate([self.data, jnp.reshape(s.data, [1, -1])], axis=0)
        return self


    def pre_append(self, s, t):
        # s, t are of Text_Embedding type
        if self.data is None:
            self.data = jnp.concatenate([jnp.reshape(s.data, [1, -1]), jnp.reshape(t.data, [1, -1])], axis=0)
        else:
            self.data = jnp.concatenate([jnp.reshape(s.data, [1, -1]), self.data, jnp.reshape(t.data, [1, -1])], axis=0)
        return self
        

class Augmented_Embedding_Squence(humn.algebraic.Augmented_State_Squence):

    def __init__(self, data):
        # data has shape (N, 2, dim)
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))


    def __len__(self):
        return self.data.shape[0]
    

class Text_Embedding(humn.algebraic.State, humn.algebraic.Action):
    
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))


    def __add__(self, a):
        return a


    def __sub__(self, augmented_state_sequence: Augmented_Embedding_Squence):
        s = augmented_state_sequence.data[-1, 0, :]
        if jnp.linalg.norm(s - STOP_EMBEDDING.data) < 1e-2:
            return Text_Embedding(jnp.zeros_like(self.data))
        
        return self


    def zero_length(self):
        return jnp.linalg.norm(self.data) < 1e-4
    

STOP_EMBEDDING = None

def set_stop_embedding(embedding):
    global STOP_EMBEDDING
    STOP_EMBEDDING = Text_Embedding(embedding)