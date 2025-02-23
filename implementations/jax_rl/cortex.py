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
    
    if not use_monte_carlo:
        # inferred_scores has shape (P, seq_len)
        _, raw_inferred_scores = model_infer(
            jnp.reshape(s, (-1, 1, s.shape[2])),
            jnp.reshape(t, (-1, t.shape[2]))
        )
        inferred_scores = jnp.reshape(raw_inferred_scores, (num_pivots, num_steps))
        
        if not use_reward:
            # replace score at pivot with 1.0
            dirac = generate_pivot_dirac_mask(pivot_indices, num_steps)
            inferred_scores = inferred_scores * (1 - dirac) + dirac

        # roll the scores
        inferred_scores = jnp.roll(inferred_scores, -1, axis=1)
        # now discount the scores
        scores = inferred_scores * step_discount_factor

        if use_reward:
            # td learning with reward
            reward_sequence =  jnp.reshape(reward_sequence, (-1))
            tiled_rewards = jnp.tile(jnp.expand_dims(reward_sequence, axis=0), (num_pivots, 1))
            scores = tiled_rewards + scores
    else:

        if use_reward:
            # exp to prevent less than zero reward
            sum_reward = jnp.sum(reward_sequence)
            scores = masks * sum_reward
        else:
            # monte carlo update, scores are the discount table
            scores = generate_score_matrix(pivot_indices, num_steps, step_discount_factor)

    return s, x, t, scores, masks


@partial(jax.jit, static_argnames=['s_dim', 'x_dim', 't_dim', 'seq_len', 'round_up_len', 'max_learning_sequence'])
def split_data(s, x, t, scores, masks, s_dim, x_dim, t_dim, seq_len, round_up_len, max_learning_sequence):
    s = jnp.pad(s, ((0, 0), (round_up_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
    x = jnp.pad(x, ((0, 0), (round_up_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
    t = jnp.pad(t, ((0, 0), (round_up_len - seq_len, 0), (0, 0)), mode="constant", constant_values=0.0)
    scores = jnp.pad(scores, ((0, 0), (round_up_len - seq_len, 0)), mode="constant", constant_values=0.0)
    masks = jnp.pad(masks, ((0, 0), (round_up_len - seq_len, 0)), mode="constant", constant_values=0.0)

    return jnp.reshape(s, (-1, max_learning_sequence, s_dim)), jnp.reshape(x, (-1, max_learning_sequence, x_dim)), jnp.reshape(t, (-1, max_learning_sequence, t_dim)), jnp.reshape(scores, (-1, max_learning_sequence)), jnp.reshape(masks, (-1, max_learning_sequence))


# arr is array of floating point
# arr is sorted in descending order
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if abs(arr[mid] - target) < 1e-6:
            # Target already exists, insert before it
            return mid
        elif arr[mid] > target:
            low = mid + 1  # Target should be inserted to the right
        else:  # arr[mid] < target
            high = mid - 1  # Target should be inserted to the left
    return low  # Return 'low' as the insertion index


class Trainer(trainer.Trainer):
    def __init__(self, total_keeping=1000, loss_alpha=0.05):
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
        self.avg_loss = jnp.zeros(1)


    def serialize(self):
        return {
            "_class_name": "Trainer",
            "total_keeping": self.total_keeping,
            "loss_alpha": self.loss_alpha
        }


    def set_model(self, model):
        self.model = model


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
            s, x, t, scores, masks = split_data(self.s[i], self.x[i], self.t[i], self.scores[i], self.masks[i], self.s[i].shape[2], self.x[i].shape[2], self.t[i].shape[2], seq_len, round_up_len, max_learning_sequence)
            current_size += batch_len * int(round_up_len / max_learning_sequence)
            current_s.append(s)
            current_x.append(x)
            current_t.append(t)
            current_scores.append(scores)
            current_masks.append(masks)

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
        return self


    def get_loss(self):
        # convert to cpu
        return self.avg_loss.item()
    

class Trainer_Online(trainer.Trainer):

    def __init__(self, loss_alpha=0.05, max_mini_batch_size=16, max_learning_sequence=16): 
        self.max_mini_batch_size = max_mini_batch_size
        self.max_learning_sequence = max_learning_sequence
        self.loss_alpha = loss_alpha
        self.avg_loss = jnp.zeros(1)


    def serialize(self):
        return {
            "_class_name": "Trainer_Online",
            "loss_alpha": self.loss_alpha,
            "max_mini_batch_size": self.max_mini_batch_size,
            "max_learning_sequence": self.max_learning_sequence
        }


    def set_model(self, model):
        self.model = model


    def accumulate_batch(self, step_discount_factor, use_action, use_reward, use_monte_carlo, path_encoding_sequence: State_Action_Sequence, pivot_indices: Pointer_Sequence, pivots: Expectation_Sequence):

        if len(path_encoding_sequence) == 0:
            return self

        cart_state_sequence = path_encoding_sequence.get_states()
        action_sequence = path_encoding_sequence.get_actions()
        reward_sequence = path_encoding_sequence.get_rewards()
        goal_sequence = pivots.get()

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

        seq_len = s.shape[1]
        round_up_len = math.ceil(seq_len / self.max_learning_sequence) * self.max_learning_sequence
        s, x, t, scores, masks = split_data(s, x, t, scores, masks, s.shape[2], x.shape[2], t.shape[2], seq_len, round_up_len, self.max_learning_sequence)
        
        for i in range(0, s.shape[0], self.max_mini_batch_size):
            loss = self.model.fit_sequence(
                s[i: i + self.max_mini_batch_size, ...],
                x[i: i + self.max_mini_batch_size, ...],
                t[i: i + self.max_mini_batch_size, ...],
                scores[i: i + self.max_mini_batch_size, ...],
                masks[i: i + self.max_mini_batch_size, ...]
            )
            self.avg_loss = (self.avg_loss * (1-self.loss_alpha) + loss * self.loss_alpha)

        return self
    

    def get_loss(self):
        # convert to cpu
        return self.avg_loss.item()


class Model(cortex_model.Model):
    
    def __init__(self, layer: int, return_action: bool, model: core.base.Model, trainer: trainer.Trainer, step_discount_factor=0.9, use_reward=False, use_monte_carlo=True):
        # if you wish to share the model, pass the index into learning and inference functions to differentiate between layers
        self.layer = layer
        self.return_action = return_action
        self.step_discount_factor = step_discount_factor
        self.use_reward = use_reward
        self.use_monte_carlo = use_monte_carlo

        self.model = model
        self.printer = None

        self.trainer = trainer
        self.trainer.set_model(self.model)


    def set_printer(self, printer):
        self.printer = printer


    def set_update_mode(self, use_monte_carlo):
        self.use_monte_carlo = use_monte_carlo


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = core.load(metadata["model"])
        if metadata["trainer"]["_class_name"] == "Trainer":
            trainer = Trainer(
                total_keeping=metadata["trainer"]["total_keeping"],
                loss_alpha=metadata["trainer"]["loss_alpha"]
            )
        else:
            trainer = Trainer_Online(
                loss_alpha=metadata["trainer"]["loss_alpha"],
                max_mini_batch_size=metadata["trainer"]["max_mini_batch_size"],
                max_learning_sequence=metadata["trainer"]["max_learning_sequence"]
            )
        return Model(
            layer=metadata["layer"], 
            return_action=metadata["return_action"], 
            model=model,
            trainer=trainer,
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
                "model": core.save(self.model),
                "trainer": self.trainer.serialize(),
                "step_discount_factor": self.step_discount_factor,
                "use_reward": self.use_reward,
                "use_monte_carlo": self.use_monte_carlo,
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
