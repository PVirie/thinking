from humn.interfaces.pathway import cortex
from typing import Tuple

import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from algebric import *
import core


def generate_mask_and_score(pivots, length, diminishing_factor=0.9, pre_steps=1):
    # pivots = jnp.array(pivots, dtype=jnp.int32)
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    
    pre_pivots = jnp.concatenate([jnp.full([pre_steps], -1, dtype=jnp.int32), pivots[:-pre_steps]], axis=0)

    masks = jnp.logical_and(pos > jnp.expand_dims(pre_pivots, axis=0), pos <= jnp.expand_dims(pivots, axis=0)).astype(jnp.float32)

    order = jnp.reshape(jnp.arange(0, -length, -1), [-1, 1]) + jnp.expand_dims(pivots, axis=0)
    scores = jnp.power(diminishing_factor, order)
    return masks, scores


class Model(cortex.Cortex_Pathway):
    
    def __init__(self, model: core.base.Model, step_discount_factor=0.9):
        super().__init__(step_discount_factor=step_discount_factor)
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


    def refresh(self):
        pass



    def infer_sub_action(self, from_state: State, expect_action: Action) -> Tuple[Action, float]:
        goal_state = from_state + expect_action
        next_action_data, score = self.model.infer(from_state.data, goal_state.data)
        return Action(next_action_data), score



    def incrementally_learn(self, path: State_Sequence, pivot_indices: Index_Sequence, pivots: State_Sequence):
        distances = jnp.arange(len(path))

        # learn to predict the next state and its probability from the current state given goal
        masks, scores = generate_mask_and_score(pivot_indices.data, len(path))

        s = jnp.tile(jnp.expand_dims(path.data, axis=1), (1, len(pivots), 1))
        x = jnp.tile(jnp.expand_dims(jnp.roll(path.data, -1, axis=0), axis=1), (1, len(pivots), 1))
        t = jnp.tile(jnp.expand_dims(pivots.data[pivot_indices.data], axis=0), (len(path), 1, 1))

        # s has shape (N, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N)
        self.model.fit(s, x, t, scores, masks)




if __name__ == "__main__":
    masks, scores = generate_mask_and_score(jnp.array([1, 3, 8]), 8)
    print(masks)
    print(scores)

    table_model = core.table.Model(4)
    model = Model(table_model)

    states = State_Sequence(jnp.eye(4, dtype=jnp.float32))
    pivot_indices, pivots = states.sample_skip(2, include_last=True)

    model.incrementally_learn(states, pivot_indices, pivots)