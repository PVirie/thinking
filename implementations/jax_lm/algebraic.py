import jax.numpy as jnp
from jax import device_put
from typing import List
import humn


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
            return Pointer_Sequence(jnp.array([s], dtype=jnp.int32))
        else:
            return Pointer_Sequence(jnp.concatenate([self.data, jnp.array([s], dtype=jnp.int32)], axis=0))

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


    def append(self, s):
        # s is of Text_Embedding type
        if self.data is None:
            return Embedding_Sequence(jnp.reshape(s.data, [1, -1]))
        else:
            return Embedding_Sequence(jnp.concatenate([self.data, jnp.reshape(s.data, [1, -1])], axis=0))


    def prepend(self, s):
        # s is of Text_Embedding type
        if self.data is None:
            return Embedding_Sequence(jnp.reshape(s.data, [1, -1]))
        else:
            return Embedding_Sequence(jnp.concatenate([jnp.reshape(s.data, [1, -1]), self.data], axis=0))
        

class Augmented_Embedding_Squence(humn.algebraic.Augmented_State_Squence):

    def __init__(self, data, stop_end=False):
        # data has shape (N, 2, dim)
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))

        self.stop_end = stop_end

    def __len__(self):
        return self.data.shape[0]
    


class State_Action(humn.algebraic.State, humn.algebraic.Action):
    
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))



class Text_Embedding(humn.algebraic.State, humn.algebraic.Action):
    
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))


    def __add__(self, a: State_Action):
        return a


    def __sub__(self, augmented_state_sequence: Augmented_Embedding_Squence):
        if augmented_state_sequence.stop_end:
            return Text_Embedding(jnp.zeros_like(self.data))
        return self


    def zero_length(self):
        return jnp.linalg.norm(self.data) < 1e-4
    