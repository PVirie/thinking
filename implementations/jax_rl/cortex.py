from humn import cortex_model, trainer
from typing import Tuple
import random
from enum import Enum

from functools import partial
import jax.numpy as jnp
import os
import json
import math

try:
    from algebraic import *
except:
    from .algebraic import *

try:
    import core
except:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    import core


@partial(jax.jit, static_argnames=['length', 'diminishing_factor'])
def generate_score_matrix(pivots, length, diminishing_factor=0.9):
    order = jnp.reshape(jnp.arange(0, -length, -1), [-1, 1]) + jnp.expand_dims(pivots, axis=0)
    scores = jnp.power(diminishing_factor, order)
    return jnp.transpose(scores)


@partial(jax.jit, static_argnames=['length', 'pre_steps'])
def generate_mask(pivots, length, pre_steps=1):
    # from states to pivots
    # pivots = jnp.array(pivots, dtype=jnp.int32)
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    pre_pivots = jnp.concatenate([jnp.full([pre_steps], -1, dtype=jnp.int32), pivots[:-pre_steps]], axis=0)
    masks = jnp.logical_and(jnp.expand_dims(pre_pivots, axis=0) <= pos, pos < jnp.expand_dims(pivots, axis=0)).astype(jnp.float32)
    return jnp.transpose(masks)


@partial(jax.jit, static_argnames=['length'])
def generate_pivot_dirac_mask(pivots, length):
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    scores = pos == jnp.expand_dims(pivots, axis=0)
    scores = scores.astype(jnp.float32)
    return jnp.transpose(scores)


@partial(jax.jit, static_argnames=['length', 'diminishing_factor', 'upper_triangle'])
def generate_geometric_matrix(length, diminishing_factor=0.9, upper_triangle=True):
    grid_x = jnp.arange(length)
    grid_y = jnp.arange(length)
    grid_x, grid_y = jnp.meshgrid(grid_x, grid_y)
    grid = jnp.abs(grid_x - grid_y)
    grid = jnp.power(diminishing_factor, grid)
    # remove triangle, keep diagonal
    if upper_triangle:
        grid = jnp.triu(grid, 0)
    else:
        grid = jnp.triu(grid, 1)
    return grid


@partial(jax.jit, static_argnames=['num_pivots', 'num_steps', 'step_discount_factor', 'model_infer', 'use_action', 'use_reward', 'use_monte_carlo'])
def prepare_data(cart_state_sequence, action_sequence, reward_sequence, goal_sequence, pivot_indices, num_pivots, num_steps, step_discount_factor, model_infer, use_action, use_reward, use_monte_carlo):

    # s has shape (P, seq_len, state_dim)
    s = jnp.tile(jnp.expand_dims(cart_state_sequence, axis=0), (num_pivots, 1, 1))

    # t has shape (P, seq_len, goal_dim)
    t = jnp.tile(jnp.expand_dims(goal_sequence, axis=1), (1, num_steps, 1))

    if use_action:
        x = jnp.tile(jnp.expand_dims(action_sequence, axis=0), (num_pivots, 1, 1))
    else:
        # roll cause weirdness at edge, must mask out the last one by make sure that the last item in the pivot_indices is len(path_encoding_sequence) - 1
        x = jnp.tile(jnp.expand_dims(jnp.roll(cart_state_sequence, -1, axis=0), axis=0), (num_pivots, 1, 1))

    # s has shape (P, seq_len, dim), a has shape (P, seq_len, dim), t has shape (P, seq_len, dim), scores has shape (P, seq_len), masks has shape (P, seq_len)
    masks = generate_mask(pivot_indices, num_steps, min(2, num_pivots))
    
    if not use_monte_carlo or use_reward:
        # inferred_scores has shape (P, seq_len)
        _, raw_inferred_scores = model_infer(
            jnp.reshape(s, (-1, 1, s.shape[2])),
            jnp.reshape(t, (-1, t.shape[2]))
        )
        inferred_scores = jnp.reshape(raw_inferred_scores, (num_pivots, num_steps))
        # replace score at pivot with 1.0
        dirac = generate_pivot_dirac_mask(pivot_indices, num_steps)
        inferred_scores = inferred_scores * (1 - dirac) + dirac
        # roll the scores
        inferred_scores = jnp.roll(inferred_scores, -1, axis=1)
        # now discount the scores
        scores = inferred_scores * step_discount_factor

        if use_reward:
            # td learning with reward
            reward_sequence = jnp.reshape(reward_sequence, (-1))
            tiled_rewards = jnp.tile(jnp.expand_dims(reward_sequence, axis=0), (num_pivots, 1))
            scores = tiled_rewards + scores
    else:
        # monte carlo update, scores are the discount table
        scores = generate_score_matrix(pivot_indices, num_steps, step_discount_factor)

    return s, x, t, scores, masks


# arr is array of floating point
# arr is sorted in descending order
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if abs(arr[mid] - target) < 1e-6:
            return -1  # Target already exists, you might choose to handle this differently
        elif arr[mid] > target:
            low = mid + 1  # Target should be inserted to the right
        else:  # arr[mid] < target
            high = mid - 1  # Target should be inserted to the left
    return low  # Return 'low' as the insertion index


class Trainer(trainer.Trainer):
    def __init__(self, model, total_keeping=1000, loss_alpha=0.05):
        self.model = model

        self.total_keeping = total_keeping
        self.total_rewards = []

        self.s = []
        self.x = []
        self.t = []
        self.scores = []
        self.masks = []

        self.step = 0
        self.epoch_batch = []

        self.loss_alpha = loss_alpha
        self.avg_loss = 0.0


    def __find_insert_index(self, total_reward):
        # find the index to insert, assume that the total_rewards is sorted in descending order
        # total_reward is a float
        # also remove the least total reward if the total_rewards exceed the limit
        # return None if the total_reward is not in the top total_keeping
        index = binary_search(self.total_rewards, total_reward)
        if index < 0:
            return None
        if self.total_keeping is not None and index >= self.total_keeping:
            return None
        if self.total_keeping is not None and len(self.total_rewards) >= self.total_keeping:
            self.total_rewards.pop()
            self.s.pop()
            self.x.pop()
            self.t.pop()
            self.scores.pop()
            self.masks.pop()
        self.total_rewards.insert(index, total_reward)
        return index


    def accumulate_batch(self, step_discount_factor, use_action, use_reward, use_monte_carlo, path_encoding_sequence: State_Action_Sequence, pivot_indices: Pointer_Sequence, pivots: Expectation_Sequence):

        if len(path_encoding_sequence) == 0:
            return self

        cart_state_sequence = path_encoding_sequence.get_states()
        action_sequence = path_encoding_sequence.get_actions()
        reward_sequence = path_encoding_sequence.get_rewards()
        goal_sequence = pivots.get()

        total_reward = jnp.sum(reward_sequence).item()
        insert_index = self.__find_insert_index(total_reward)
        if insert_index is None:
            return self

        s, x, t, scores, masks = prepare_data(
            cart_state_sequence,
            action_sequence,
            reward_sequence,
            goal_sequence,
            pivot_indices.data,
            len(pivots),
            len(path_encoding_sequence),
            step_discount_factor,
            self.model.infer,
            use_action,
            use_reward,
            use_monte_carlo
        )

        if insert_index >= len(self.total_rewards):
            self.s.append(s)
            self.x.append(x)
            self.t.append(t)
            self.scores.append(scores)
            self.masks.append(masks)
        else:
            self.s.insert(insert_index, s)
            self.x.insert(insert_index, x)
            self.t.insert(insert_index, t)
            self.scores.insert(insert_index, scores)
            self.masks.insert(insert_index, masks)

        return self


    def clear_batch(self):
        self.total_rewards = []
        self.s = []
        self.x = []
        self.t = []
        self.scores = []
        self.masks = []
        return self
    

    def prepare_batch(self, max_mini_batch_size, max_learning_sequence=16):
        # try grouping batch with the same sequence length
        self.epoch_batch = []
        current_size = 0
        current_s = []
        current_x = []
        current_t = []
        current_scores = []
        current_masks = []
        for i in range(len(self.s)):
            batch_len = self.s[i].shape[0]
            seq_len = self.s[i].shape[1]
            # split into max learning sequence size
            round_up_len = math.ceil(seq_len / max_learning_sequence) * max_learning_sequence

            s = jnp.pad(self.s[i], ((0, 0), (round_up_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
            x = jnp.pad(self.x[i], ((0, 0), (round_up_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
            t = jnp.pad(self.t[i], ((0, 0), (round_up_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
            scores = jnp.pad(self.scores[i], ((0, 0), (round_up_len - seq_len, 0)), mode="constant", constant_values=0.0)
            masks = jnp.pad(self.masks[i], ((0, 0), (round_up_len - seq_len, 0)), mode="constant", constant_values=0.0)

            current_size += batch_len * int(round_up_len / max_learning_sequence)
            current_s.append(jnp.reshape(s, (-1, max_learning_sequence, s.shape[2])))
            current_x.append(jnp.reshape(x, (-1, max_learning_sequence, x.shape[2])))
            current_t.append(jnp.reshape(t, (-1, max_learning_sequence, t.shape[2])))
            current_scores.append(jnp.reshape(scores, (-1, max_learning_sequence)))
            current_masks.append(jnp.reshape(masks, (-1, max_learning_sequence)))

            if current_size > max_mini_batch_size:
                S = jnp.concatenate(current_s, axis=0)
                X = jnp.concatenate(current_x, axis=0)
                T = jnp.concatenate(current_t, axis=0)
                Scores = jnp.concatenate(current_scores, axis=0)
                Masks = jnp.concatenate(current_masks, axis=0)

                for j in range(0, current_size - max_mini_batch_size, max_mini_batch_size):
                    self.epoch_batch.append((
                        S[j:j + max_mini_batch_size],
                        X[j:j + max_mini_batch_size],
                        T[j:j + max_mini_batch_size],
                        Scores[j:j + max_mini_batch_size],
                        Masks[j:j + max_mini_batch_size]
                    ))

                residual = current_size % max_mini_batch_size

                # add residual
                if residual > 0:
                    current_size = residual
                    current_s = [S[-residual:]]
                    current_x = [X[-residual:]]
                    current_t = [T[-residual:]]
                    current_scores = [Scores[-residual:]]
                    current_masks = [Masks[-residual:]]
                else:
                    current_size = 0
                    current_s = []
                    current_x = []
                    current_t = []
                    current_scores = []
                    current_masks = []
    

        if current_size > 0:
            S = jnp.concatenate(current_s, axis=0)
            X = jnp.concatenate(current_x, axis=0)
            T = jnp.concatenate(current_t, axis=0)
            Scores = jnp.concatenate(current_scores, axis=0)
            Masks = jnp.concatenate(current_masks, axis=0)
            self.epoch_batch.append((
                S,
                X,
                T,
                Scores,
                Masks
            ))

        # shuffle
        random.shuffle(self.epoch_batch)
        

    def step_update(self):
        minibatch = self.epoch_batch[self.step % len(self.epoch_batch)]
        self.step += 1
        loss = self.model.fit_sequence(minibatch[0], minibatch[1], minibatch[2], minibatch[3], minibatch[4])
        self.avg_loss = (self.avg_loss * (1-self.loss_alpha) + loss * self.loss_alpha)
        return self.avg_loss


class Model(cortex_model.Model):
    
    def __init__(self, layer: int, return_action: bool, model: core.base.Model, step_discount_factor=0.9, use_reward=False, use_monte_carlo=True, num_items_to_keep=None):
        # if you wish to share the model, pass the index into learning and inference functions to differentiate between layers
        self.layer = layer
        self.return_action = return_action
        self.step_discount_factor = step_discount_factor
        self.use_reward = use_reward
        self.use_monte_carlo = use_monte_carlo

        self.model = model
        self.printer = None

        self.trainer = Trainer(self.model, total_keeping=num_items_to_keep)


    def set_printer(self, printer):
        self.printer = printer


    def set_update_mode(self, use_monte_carlo):
        self.use_monte_carlo = use_monte_carlo


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = core.load(metadata["model"])
        return Model(
            layer=metadata["layer"], 
            return_action=metadata["return_action"], 
            model=model, 
            step_discount_factor=metadata["step_discount_factor"], 
            use_reward=metadata["use_reward"], 
            use_monte_carlo=metadata["use_monte_carlo"]
        )
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "layer": self.layer,
                "return_action": self.return_action,
                "step_discount_factor": self.step_discount_factor,
                "use_reward": self.use_reward,
                "use_monte_carlo": self.use_monte_carlo,
                "model": core.save(self.model)
            }, f)


    def incrementally_learn(self, path_encoding_sequence: State_Action_Sequence, pivot_indices: Pointer_Sequence, pivots: Expectation_Sequence) -> Trainer:
        # learn to predict the next state and its probability from the current state given goal
        return self.trainer.accumulate_batch(self.step_discount_factor, self.return_action, self.use_reward, self.use_monte_carlo, path_encoding_sequence, pivot_indices, pivots)


    def infer_sub_action(self, from_encoding_sequence: State, expect_action: Action) -> Action:
        next_action_data, score = self.model.infer(
            jnp.reshape(from_encoding_sequence.data, (1, 1, -1)), 
            jnp.reshape(expect_action.data, (1, -1))
            )
        if self.return_action:
            a = Action(next_action_data[0])
        else:
            a = Expectation(next_action_data[0])
        if self.printer is not None:
            self.printer(from_encoding_sequence + a)
        return a


if __name__ == "__main__":
    import jax

    masks = generate_mask(jnp.array([0, 4, 7]), 8, 2)
    print(masks)

    scores = generate_score_matrix(jnp.array([0, 4, 7]), 8, 0.9)
    print(scores)
    
    dirac = generate_pivot_dirac_mask(jnp.array([0, 4, 7]), 8)
    print(dirac)

    grid = generate_geometric_matrix(8, 0.9, upper_triangle=True)
    print(grid)
