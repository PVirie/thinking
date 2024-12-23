"""
Table model for one-hot graph only (for model checking)
Author: P.Virie

"""

import jax
import jax.numpy as jnp
import os
import itertools
from functools import partial

try:
    from . import base
except:
    import base


class Model(base.Model):

    def __init__(self, dims, context_length):
        super().__init__("model", "table")

        # if dims is an integer, then dims is the number of dimensions
        self.dims = dims
        if isinstance(dims, int):
            self.input_dims = dims
            self.next_state_dims = dims
            self.target_dims = dims
        else:
            self.input_dims = dims[0]
            self.next_state_dims = dims[1]
            self.target_dims = dims[2]

        self.context_length = context_length

        hidden_size = pow(dims, context_length + 1)
        
        # make one hot all combination table of size [hidden_size, dims * (context_length + 1)] 
        # [[0, 0, 1, ..., 0, 0, 1], [0, 0, 1, ..., 0, 1, 0], ..., [1, 0, 0, ..., 1, 0, 0]]
        eye = jnp.eye(dims, dtype=jnp.float32)
        sub_keys = []
        for tuple in itertools.product(range(dims), repeat=context_length + 1):
            sub_key = []
            for i in tuple:
                sub_key.append(eye[i])
            sub_keys.append(jnp.concatenate(sub_key, axis=0))
        self.key = jnp.stack(sub_keys, axis=0)

        self.score = jnp.zeros([hidden_size, 1], jnp.float32)
        self.value = jnp.zeros([hidden_size, dims], jnp.float32)


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "dims": self.dims,
            "context_length": self.context_length
        }


    def save(self, path):
        jnp.save(os.path.join(path, "score.npy"), self.score)
        jnp.save(os.path.join(path, "value.npy"), self.value)
        self.is_updated = False

    def load(self, path):
        self.score = jnp.load(os.path.join(path, "score.npy"))
        self.value = jnp.load(os.path.join(path, "value.npy"))
        self.is_updated = False
    

    def fit(self, s, x, t, scores, masks=None, context=None):
        # s has shape (N, context_length, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)

        if masks is None:
            masks = jnp.ones(scores.shape)

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))
        x = jnp.reshape(x, (-1, self.next_state_dims))

        queries = jnp.concatenate([jnp.reshape(s, (-1, self.input_dims * self.context_length)), t], axis=-1)
        batch = jnp.reshape(queries, (-1, self.input_dims * self.context_length + self.target_dims))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        # logits has shape [N, len(key)]
        # find max key for each batch
        argmax_logits = jnp.argmax(logits, axis=1)

        # update indices where scores are higher than current scores
        update_indices = (scores > self.score[argmax_logits]).astype(jnp.float32) * masks
        score_updates = (1 - update_indices) * self.score[argmax_logits] + update_indices * scores
        # update score at argmax_logits with updates
        self.score = self.score.at[argmax_logits].set(score_updates)

        # only select values that are argmax_logits
        value_updates = (1 - update_indices) * self.value[argmax_logits] + update_indices * x
        # update values at argmax_logits with value_updates
        self.value = self.value.at[argmax_logits].set(value_updates)

        self.is_updated = True
        return jnp.mean(score_updates)


    def fit_sequence(self, s, x, t, scores, masks=None, context=None):
        # s has shape (N, seq_len, dim), x has shape (N, seq_len, dim), t has shape (N, seq_len, dim), scores has shape (N, seq_len), masks has shape (N, seq_len)
        # seq_len = learning_length + context_length - 1

        if masks is None:
            masks = jnp.ones(scores.shape)

        # first pad with zeros
        sp = jnp.pad(s, ((0, 0), (self.context_length - 1, 0), (0, 0)), mode='constant', constant_values=0)

        # then unroll by shift and tile
        # unrolled_s has shape (N, seq_len, context_length, dim)
        unrolled_s = jnp.stack([sp[:, i:i + self.context_length] for i in range(sp.shape[1] - self.context_length + 1)], axis=1)
                
        return self.fit(
            jnp.reshape(unrolled_s, (-1, self.context_length, self.input_dims)),
            jnp.reshape(x, (-1, self.next_state_dims)), 
            jnp.reshape(t, (-1, self.target_dims)), 
            jnp.reshape(scores, (-1)), 
            jnp.reshape(masks, (-1)), context)

    
    def infer(self, s, t, context=None):
        # s has shape (N, context_length, dim), t has shape (N, dim)

        if s.shape[1] < self.context_length:
            # pad input
            s = jnp.pad(s, ((0, 0), (0, self.context_length - s.shape[1]), (0, 0)), mode='constant', constant_values=0)
        elif s.shape[1] > self.context_length:
            s = s[:, -self.context_length:, :]

        # for simple model only use the last state
        queries = jnp.concatenate([jnp.reshape(s, (-1, self.input_dims * self.context_length)), t], axis=-1)
        batch = jnp.reshape(queries, (-1, self.input_dims * self.context_length + self.target_dims))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        # logits has shape [N, len(key)]
        # find max key for each batch
        argmax_logits = jnp.argmax(logits, axis=1)
        # return best score, value

        best_score = jnp.reshape(self.score[argmax_logits], [queries.shape[0]])
        best_value = jnp.reshape(self.value[argmax_logits], [queries.shape[0], -1])

        return best_value, best_score



if __name__ == "__main__":
    model = Model(4, 2)
    print(model.key)

    eye = jnp.eye(4, dtype=jnp.float32)
    S = jnp.array([[eye[0, :], eye[1, :], eye[2, :], eye[3, :]]])
    X = jnp.array([[eye[3, :], eye[3, :], eye[1, :], eye[1, :]]])
    T = jnp.array([[eye[2, :], eye[2, :], eye[3, :], eye[3, :]]])
    scores = jnp.array([[0.72, 0.81, 0.9, 1.0]])

    for i in range(1000):
        loss = model.fit_sequence(S, X, T, scores)

    s = jnp.array([[eye[0, :], eye[1, :]], [eye[1, :], eye[2, :]]])
    t = jnp.array([eye[2, :], eye[3, :]])

    value, score = model.infer(s, t)
    print("Loss:", loss)
    print("Score:", score)
    print("Value:", value)
    
    # s = jnp.array([[eye[0, :]]])
    # t = jnp.array([eye[3, :]])

    # value, score = model.infer(s, t)
    # print("Loss:", loss)
    # print("Score:", score)
    # print("Value:", value)

    # s = jnp.array([[eye[0, :], eye[1, :]], [eye[1, :], eye[2, :]]])
    # x = jnp.array([eye[3, :], eye[0, :]])
    # t = jnp.array([eye[2, :], eye[2, :]])
    # scores = jnp.array([0.81, 0.9])

    # for i in range(1000):
    #     loss = model.fit(s, x, t, scores)

    # value, score = model.infer(s, t)
    # print("Loss:", loss)
    # print("Score:", score)
    # print("Value:", value)