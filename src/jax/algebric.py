import jax.numpy as jnp
from jax import device_put
from typing import List
import humn
import os
import math

class Action(humn.algebraic.Action):
    def __init__(self, data):
        # if data is jax array, then it is already on device
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))

    def __add__(self, s): 
        return s + self
    
    def zero_length(self):
        return jnp.linalg.norm(self.data) < 1e-4
    

class State(humn.algebraic.State):
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))

    def __add__(self, a):
        return State(self.data + a.data)


    def __sub__(self, s):
        return Action(self.data - s.data)


    def __eq__(self, s):
        # implement ==
        # test norm within 1e-4
        return jnp.linalg.norm(self.data - s.data) < 1e-4



class Pointer_Sequence(humn.algebraic.Pointer_Sequence):
    def __init__(self, indices = []):
        self.data = jnp.array(indices, dtype=jnp.int32)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, s):
        self.data = self.data.at[i].set(s)

    def append(self, s):
        self.data = jnp.concatenate([self.data, jnp.array([s], dtype=jnp.int32)], axis=0)




class State_Sequence(humn.algebraic.State_Sequence):
    def __init__(self, states):
        if isinstance(states, List):
            self.data = jnp.array([s.data for s in states])
        elif isinstance(states, jnp.ndarray):
            self.data = states
        else:
            self.data = device_put(jnp.array(states, jnp.float32))



    @staticmethod
    def load(path):
        # load jax array from path
        return State_Sequence(jnp.load(path + ".npy"))


    @staticmethod
    def save(self, path):
        # save jax array to path
        jnp.save(path + ".npy", self.data)



    def __getitem__(self, i):
        if isinstance(i, State):
            return self.match_index(i)
        elif isinstance(i, slice):
            return State_Sequence(self.data[i])
        elif isinstance(i, Pointer_Sequence):
            return State_Sequence(self.data[i.data])
        else:
            return State(self.data[i, :])


    def __setitem__(self, i, s):
        if isinstance(i, State):
            loc = self.match_index(i)
            self.data = self.data.at[loc].set(s.data)
        else:
            self.data = self.data.at[i].set(s.data)


    def __len__(self):
        return self.data.shape[0]


    def append(self, s):
        self.data = jnp.concatenate([self.data, jnp.expand_dims(s.data, axis=0)], axis=0)


    def sample_skip(self, n):
        if n == math.inf:
        # if n is math.inf, return first and last indices
            return Pointer_Sequence([0, len(self) - 1]), State_Sequence(self.data[[0, len(self) - 1], :])

        # return indices and states
        indices = [0]
        i = n
        while i < len(self):
            indices.append(i)
            i += n

        if (indices[len(indices) - 1] != len(self) - 1):
            indices.append(len(self) - 1)

        indice_sequence = Pointer_Sequence(indices)
        states = self.data[indice_sequence.data]
        return indice_sequence, State_Sequence(states)
    

    def match_index(self, s):
        # find index of state s
        return jnp.argmin(jnp.linalg.norm(self.data - s.data, axis=1))
    

    def generate_subsequence(self, indices: Pointer_Sequence):
        return State_Sequence(self.data[indices.data])



class Augmented_State_Squence(humn.algebraic.Augmented_State_Squence):
    
    def __init__(self, data):
        # data has shape (N, 2, dim)
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))


    def __add__(self, a):
        # return state self.data[-1, 0, dim] + a.data
        return State(self.data[-1, 0, :]) + a
    

    def __len__(self):
        return self.data.shape[0]
    

    def __getitem__(self, i):
        if isinstance(i, slice):
            # if i is a slice
            return Augmented_State_Squence(self.data[i])
        elif isinstance(i, Pointer_Sequence):
            # if i is an index sequence
            return Augmented_State_Squence(self.data[i.data])
        else:
            # if i is an integer
            return State(self.data[i, 0, :])
