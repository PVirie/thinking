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

    @staticmethod
    def load(path):
        pass

    def save(self, path):
        pass

class Index_Sequence(humn.Index_Sequence):
    def __init__(self, indices = []):
        self.list = indices

    def __getitem__(self, i):
        return self.list[i]

    def __setitem__(self, i, s):
        self.list[i] = s

    def append(self, s):
        self.list.append(s)




class State_Sequence(humn.State_Sequence):
    def __init__(self, states):
        self.start = states[0]
        self.actions = [states[i] - states[i - 1] for i in range(1, len(states))]


    def __getitem__(self, i):
        # check type if i is a State then return best match indice
        if isinstance(i, State):
            return self.match(i)

        if i == 0:
            return self.start
        s = self.start
        for a in self.actions[:i]:
            s += a
        return s

    def __setitem__(self, i, s):
        if i == 0:
            self.start = s
        else:
            self.actions[i - 1] = s - self[i - 1]

    def __delitem__(self, i):
        if i == 0:
            self.start = self[1]
            self.actions.pop(0)
        self.actions.pop(i)

    def __len__(self):
        return len(self.actions) + 1

    def append(self, s):
        self.actions.append(s - self[-1])

    def unroll(self):
        states = []
        s = self.start
        states.append(s)
        for a in self.actions:
            s += a
            states.append(s)
        return states
    
    
    def generate_subsequence(self, indices: Index_Sequence):
        unrolled = self.unroll()
        return State_Sequence([unrolled[i] for i in indices[1:]])


    def match(self, s):
        min_dist = 1e6
        min_i = 0
        for i, s2 in enumerate(self.unroll()):
            dist = s2 - s
            if dist.norm() < min_dist:
                min_dist = dist.norm()
                min_i = i
        return min_i