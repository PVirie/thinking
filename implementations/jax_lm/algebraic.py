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


        

class Augmented_Embedding_Squence(humn.algebraic.Augmented_State_Squence):

    def __init__(self, data):
        # data has shape (N, dim)
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))

    def __len__(self):
        return self.data.shape[0]
    


class Augmented_Text_Embedding(humn.algebraic.State, humn.algebraic.Action):
    
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))


    def __add__(self, a):
        if isinstance(a, Augmented_Text_Embedding):
            max_index = jnp.argmax(a.data[:-1])
            a_data = jnp.zeros_like(a.data).at[max_index].set(100)
            a_data = a_data.at[-1].set(100 if a.data[-1] > 50 else 0)
            return Augmented_Text_Embedding(a_data)
        else:
            return a


class Stop_Embedding(humn.algebraic.State, humn.algebraic.Action):
    
    def __init__(self):
        pass

    def zero_length(self):
        return True


class Text_Embedding(humn.algebraic.State, humn.algebraic.Action):
    
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))


    def __add__(self, a):
        if isinstance(a, Augmented_Text_Embedding):
            max_index = jnp.argmax(a.data[:-1])
            a_data = jnp.zeros_like(a.data).at[max_index].set(100)
            a_data = a_data.at[-1].set(100 if a.data[-1] > 50 else 0)
            return Augmented_Text_Embedding(a_data)
        else:
            return a


    def __sub__(self, augmented_state_sequence: Augmented_Embedding_Squence):
        if augmented_state_sequence.data[-1, -1] > 50:
            return Stop_Embedding()
        return self


    def zero_length(self):
        return False
    