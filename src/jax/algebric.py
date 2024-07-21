import jax.numpy as jnp
from jax import device_put
from typing import List
import humn

class Action(humn.Action):
    def __init__(self, data):
        # if data is jax array, then it is already on device
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))

    def __add__(self, s): 
        return s + self
    
    def norm(self):
        return jnp.linalg.norm(self.data)
    

class State(humn.State):
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
        return (self.data - s.data).norm() < 1e-4


    @staticmethod
    def load(path):
        # load jax array from path
        with open(path, "rb") as f:
            data = jnp.load(f)
        return State(data)


    def save(self, path):
        # save jax array to path
        with open(path, "wb") as f:
            jnp.save(f, self.data)



class Index_Sequence(humn.Index_Sequence):
    def __init__(self, indices = []):
        self.data = jnp.array(indices, dtype=jnp.int32)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, s):
        self.data = self.data.at[i].set(s)

    def append(self, s):
        self.data = jnp.concatenate([self.data, jnp.array([s], dtype=jnp.int32)], axis=0)




class State_Sequence(humn.State_Sequence):
    def __init__(self, states):
        if isinstance(states, List):
            self.data = jnp.array([s.data for s in states])
        elif isinstance(states, jnp.ndarray):
            self.data = states
        else:
            self.data = device_put(jnp.array(states, jnp.float32))


    def __getitem__(self, i):
        if isinstance(i, State):
            return self.match_index(i)
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


    def sample_skip(self, n, include_last=False):
        # return indices and states
        indices = Index_Sequence()
        for i in range(n, len(self), n):
            indices.append(i)
        if include_last and i != len(self) - 1:
            indices.append(len(self) - 1)

        states = self.data[indices.data]
        return indices, State_Sequence(states)
    

    def match_index(self, s):
        # find index of state s
        return jnp.argmin(jnp.linalg.norm(self.data - s.data, axis=1))
    

    def generate_subsequence(self, indices: Index_Sequence):
        return State_Sequence(self.data[indices.data])
