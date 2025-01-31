from humn import cortex_model, trainer
from typing import Tuple
import random

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


def generate_mask_and_score(pivots, length, diminishing_factor=0.9, pre_steps=1):
    # from states to pivots
    # pivots = jnp.array(pivots, dtype=jnp.int32)
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    pre_pivots = jnp.concatenate([jnp.full([pre_steps], -1, dtype=jnp.int32), pivots[:-pre_steps]], axis=0)
    # unlike other tasks, RL task is the action to get to those states, so we need to count the self
    masks = jnp.logical_and(jnp.expand_dims(pre_pivots, axis=0) < pos, pos <= jnp.expand_dims(pivots, axis=0)).astype(jnp.float32)
    order = jnp.reshape(jnp.arange(0, -length, -1), [-1, 1]) + jnp.expand_dims(pivots, axis=0)

    scores = jnp.power(diminishing_factor, order)
    return jnp.transpose(masks), jnp.transpose(scores)


class Trainer(trainer.Trainer):
    def __init__(self, model, loss_alpha=0.05):
        self.model = model

        self.s = []
        self.x = []
        self.t = []
        self.scores = []
        self.masks = []

        self.step = 0
        self.epoch_batch = []

        self.loss_alpha = loss_alpha
        self.avg_loss = 0.0


    def accumulate_batch(self, step_discount_factor, use_action, continuous_reward, path_encoding_sequence: State_Action_Sequence, pivot_indices: Pointer_Sequence, pivots: Expectation_Sequence):

        if len(path_encoding_sequence) == 0:
            return self

        cart_state_sequence = path_encoding_sequence.get_states()
        if continuous_reward:
            # pivots has the same shape as path_encoding_sequence
            s = jnp.expand_dims(cart_state_sequence, axis=0)

            if use_action:
                action_sequence = path_encoding_sequence.get_actions()
                x = jnp.expand_dims(action_sequence, axis=0)
            else:
                x = jnp.expand_dims(jnp.roll(cart_state_sequence, -1, axis=0), axis=0)

            expectation_sequence = pivots.get()
            t_scores = jnp.expand_dims(expectation_sequence, axis=0)
            t = t_scores[:, :, 1:]
            scores = t_scores[:, :, 0]
            masks = jnp.ones_like(scores)
        else:
            s = jnp.tile(jnp.expand_dims(cart_state_sequence, axis=0), (len(pivots), 1, 1))

            if use_action:
                action_sequence = path_encoding_sequence.get_actions()
                x = jnp.tile(jnp.expand_dims(action_sequence, axis=0), (len(pivots), 1, 1))
            else:
                x = jnp.tile(jnp.expand_dims(jnp.roll(cart_state_sequence, -1, axis=0), axis=0), (len(pivots), 1, 1))

            # s has shape (P, seq_len, dim), a has shape (P, seq_len, dim), t has shape (P, seq_len, dim), scores has shape (P, seq_len), masks has shape (P, seq_len)
            masks, scores = generate_mask_and_score(pivot_indices.data, len(path_encoding_sequence), step_discount_factor, min(2, pivot_indices.data.shape[0]))
            
            goal_sequence = pivots.get_goals()
            # t has shape (P, seq_len, goal_dim)
            t = jnp.tile(jnp.expand_dims(goal_sequence, axis=1), (1, len(path_encoding_sequence), 1))


        self.s.append(s)
        self.x.append(x)
        self.t.append(t)
        self.scores.append(scores)
        self.masks.append(masks)

        return self


    def clear_batch(self):
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
    
    def __init__(self, layer: int, return_action: bool, continuous_reward: bool, model: core.base.Model, step_discount_factor=0.9):
        # if you wish to share the model, pass the index into learning and inference functions to differentiate between layers
        self.layer = layer
        self.return_action = return_action
        self.continuous_reward = continuous_reward
        self.step_discount_factor = step_discount_factor
        self.model = model
        self.printer = None

        self.trainer = Trainer(self.model)


    def set_printer(self, printer):
        self.printer = printer


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = core.load(metadata["model"])
        return Model(layer=metadata["layer"], return_action=metadata["return_action"], continuous_reward=metadata["continuous_reward"], model=model, step_discount_factor=metadata["step_discount_factor"])
                                                              

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "layer": self.layer,
                "return_action": self.return_action,
                "continuous_reward": self.continuous_reward,
                "step_discount_factor": self.step_discount_factor,
                "model": core.save(self.model)
            }, f)


    def incrementally_learn(self, path_encoding_sequence: State_Action_Sequence, pivot_indices: Pointer_Sequence, pivots: Expectation_Sequence) -> Trainer:
        # learn to predict the next state and its probability from the current state given goal
        return self.trainer.accumulate_batch(self.step_discount_factor, self.return_action, self.continuous_reward, path_encoding_sequence, pivot_indices, pivots)


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

    masks, scores = generate_mask_and_score(jnp.array([0, 4, 7]), 8, 0.9, 2)
    print(masks)
    print(scores)
