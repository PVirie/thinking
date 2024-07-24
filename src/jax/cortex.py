from humn import cortex_model
from typing import Tuple

import os
import json

if __name__ == "__main__":
    from algebric import *
else:
    from .algebric import *

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import core


def generate_mask_and_score(pivots, length, diminishing_factor=0.9, pre_steps=1):
    # pivots = jnp.array(pivots, dtype=jnp.int32)
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    
    pre_pivots = jnp.concatenate([jnp.full([pre_steps], -1, dtype=jnp.int32), pivots[:-pre_steps]], axis=0)

    masks = jnp.logical_and(pos > jnp.expand_dims(pre_pivots, axis=0), pos <= jnp.expand_dims(pivots, axis=0)).astype(jnp.float32)

    order = jnp.reshape(jnp.arange(0, -length, -1), [-1, 1]) + jnp.expand_dims(pivots, axis=0)
    scores = jnp.power(diminishing_factor, order)
    return masks, scores


class Model(cortex_model.Model):
    
    def __init__(self, model: core.base.Model, step_discount_factor=0.9):
        self.step_discount_factor = step_discount_factor
        self.model = model


    @staticmethod
    def load(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        model = core.load(os.path.join(path, "model"))
        return Model(model, step_discount_factor=metadata["step_discount_factor"])
                                                              

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        core.save(os.path.join(path, "model"), self.model)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({"step_discount_factor": self.step_discount_factor}, f)



    def incrementally_learn(self, path_encoding_sequence: Augmented_State_Squence, pivot_indices: Index_Sequence, pivots: State_Sequence) -> float:

        distances = jnp.arange(len(path_encoding_sequence))

        # learn to predict the next state and its probability from the current state given goal
        masks, scores = generate_mask_and_score(pivot_indices.data, len(path_encoding_sequence))

        pivots = path_encoding_sequence[pivot_indices].data[:, 0, :]
        sequence_data = path_encoding_sequence.data[:, 0, :]

        s = jnp.tile(jnp.expand_dims(sequence_data, axis=1), (1, len(pivots), 1))
        x = jnp.tile(jnp.expand_dims(jnp.roll(sequence_data, -1, axis=0), axis=1), (1, len(pivots), 1))
        a = x - s
        t = jnp.tile(jnp.expand_dims(pivots, axis=0), (len(path_encoding_sequence), 1, 1))

        # s has shape (N, dim), a has shape (N, dim), t has shape (N, dim), scores has shape (N)
        return self.model.fit(s, a, t, scores, masks)



    def infer_sub_action(self, from_encoding_sequence: Augmented_State_Squence, expect_action: Action) -> Action:
        goal_state = from_encoding_sequence + expect_action
        next_action_data, score = self.model.infer(
            jnp.expand_dims(from_encoding_sequence.data[-1, 0, :], axis=0), 
            jnp.expand_dims(goal_state.data, axis=0)
            )
        return Action(next_action_data[0])





if __name__ == "__main__":
    import jax

    masks, scores = generate_mask_and_score(jnp.array([1, 3, 8]), 8)
    print(masks)
    print(scores)

    table_model = core.table.Model(4)
    model = Model(table_model)

    states = Augmented_State_Squence(jax.random.normal(jax.random.PRNGKey(0), (10, 2, 4)))

    model.incrementally_learn(states, Index_Sequence([5, 9]))