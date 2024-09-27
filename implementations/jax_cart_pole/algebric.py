import jax.numpy as jnp
import jax.random
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
    
    def zero_length(self):
        return jnp.linalg.norm(self.data) < 1e-4
    

# Pivots
class Expectation(humn.algebraic.State):
    def __init__(self, data):
        # if data is jax array, then it is already on device
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))
    

class Cart_State(humn.algebraic.State, humn.algebraic.Augmented_State_Squence):
    
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))



class Pointer_Sequence(humn.algebraic.Pointer_Sequence):
    def __init__(self, indices = []):
        self.data = jnp.array(indices, dtype=jnp.int32)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, s):
        self.data = self.data.at[i].set(s)

    def append(self, s):
        self.data = jnp.concatenate([self.data, jnp.array([s], dtype=jnp.int32)], axis=0)


class State_Action_Sequence(humn.algebraic.State_Sequence):
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))


class Expectation_Sequence(humn.algebraic.State_Sequence):
    def __init__(self, data):
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))
