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
class Expectation(humn.algebraic.Action):
    def __init__(self, data):
        # if data is jax array, then it is already on device
        if isinstance(data, jnp.ndarray):
            self.data = data
        else:
            self.data = device_put(jnp.array(data, jnp.float32))
    
    def zero_length(self):
        return False


class State(humn.algebraic.State, humn.algebraic.Augmented_State_Squence):
    
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

    def __len__(self):
        return self.data.shape[0]


class State_Action_Sequence(humn.algebraic.State_Sequence):
    def __init__(self, state_data, action_data, reward_data):
        if not isinstance(state_data, jnp.ndarray):
            state_data = device_put(jnp.array(state_data, jnp.float32))

        if not isinstance(action_data, jnp.ndarray):
            action_data = device_put(jnp.array(action_data, jnp.float32))

        if not isinstance(reward_data, jnp.ndarray):
            reward_data = device_put(jnp.array(reward_data, jnp.float32))

        self.state_dim = state_data.shape[1]
        self.action_dim = action_data.shape[1]
        self.data = jnp.concatenate([state_data, action_data, reward_data], axis=1)

    def get_states(self):
        return self.data[:, :self.state_dim]
    
    def get_actions(self):
        return self.data[:, self.state_dim:self.state_dim+self.action_dim]
    
    def get_rewards(self):
        return self.data[:, -1:]

    def __len__(self):
        return self.data.shape[0]


class Expectation_Sequence(humn.algebraic.State_Sequence):
    def __init__(self, goal_data):
        if not isinstance(goal_data, jnp.ndarray):
            goal_data = device_put(jnp.array(goal_data, jnp.float32))
        
        self.data = goal_data

    def get(self):
        return self.data
    
    def __len__(self):
        return self.data.shape[0]