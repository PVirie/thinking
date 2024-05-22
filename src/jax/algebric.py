import jax.numpy as jnp
from jax import device_put

import humn

class Action(humn.Action):
    def __init__(self, data):
        # if data is jax array, then it is already on device
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))

    

class State(humn.State):
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))

    def __add__(self, a): 
        # simple state, just replace with a
        return State(a.data)

    def __sub__(self, s):
        # simple state, just replace with self.data
        return Action(self.data)

    # implement ==
    def __eq__(self, s):
        # test norm within 1e-4
        return jnp.linalg.norm(self.data - s.data) < 1e-4


class Index_Sequence(humn.Index_Sequence):
    def __init__(self):
        self.list = []

    def __getitem__(self, i):
        return self.list[i]

    def __setitem__(self, i, s):
        self.list[i] = s

    def append(self, s):
        self.list.append(s)


class State_Sequence(humn.State_Sequence):
    def __init__(self, start, actions):
        super().__init__(start, actions)