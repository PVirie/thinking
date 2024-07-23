import jax.random
import jax
import jax.numpy as jnp
import os

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import base


def compute_error(M, Y, L, W):
    # trace ( [Y - L W] M [Y - L W]^T )
    E = Y - jnp.matmul(L, W)
    return jnp.trace(jnp.matmul(E, jnp.matmul(M, E.T)))

class Model(base.Model):

    def __init__(self, hidden_size, dims):
        self.class_name = "linear"
        self.hidden_size = hidden_size
        self.input_dims = dims

        self.key = jax.random.normal(jax.random.PRNGKey(0), (hidden_size, dims * 2))*0.01
        self.score = jnp.zeros([hidden_size, 1], jnp.float32)
        self.value = jnp.zeros([hidden_size, dims], jnp.float32)


    def get_class_parameters(self):
        return {"class_name": self.class_name, "input_dims": self.input_dims, "hidden_size": self.hidden_size}


    def save(self, path):
        jnp.save(os.path.join(path, "key.npy"), self.key)
        jnp.save(os.path.join(path, "score.npy"), self.score)
        jnp.save(os.path.join(path, "value.npy"), self.value)


    @staticmethod
    def load(path, class_parameters):
        model = Model(class_parameters["input_dims"])
        model.key = jnp.load(os.path.join(path, "key.npy"))
        model.score = jnp.load(os.path.join(path, "score.npy"))
        model.value = jnp.load(os.path.join(path, "value.npy"))
        return model


    def fit(self, s, x, t, scores, masks=1.0):
        # s has shape (N, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))
        x = jnp.reshape(x, (-1, self.input_dims))

        queries = jnp.concatenate([s, t], axis=-1)
        batch = jnp.reshape(queries, (-1, self.input_dims * 2))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        
        best_value = jnp.matmul(logits, self.value)
        best_score = jnp.matmul(logits, self.score)

        update_indices = (scores > best_score) * masks

        error_v = compute_error(update_indices, x, best_value, self.value)
        error_s = compute_error(update_indices, scores, best_score, self.score)


        # # score_updates = (1 - update_indices) * best_score + update_indices * scores
        # self.score = 0.95*self.score + 0.05*score_updates

        # # value_updates = (1 - update_indices) * best_value + update_indices * x
        # self.value = 0.95*self.value + 0.05*value_updates

        return jnp.mean(score_updates)


    def infer(self, s, t):
        # s has shape (N, dim), t has shape (N, dim)

        # for simple model only use the last state
        queries = jnp.concatenate([s, t], axis=-1)
        batch = jnp.reshape(queries, (-1, self.input_dims * 2))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))

        best_value = jnp.matmul(logits, self.value)
        best_score = jnp.matmul(logits, self.score)

        return best_value, best_score




if __name__ == "__main__":
    model = Model(8, 4)
    print(model.key)

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :]])
    t = jnp.array([eye[3, :], eye[3, :]])

    model.fit(s, x, t, jnp.array([1, 0]))
    score, value = model.infer(s, t)
    print(score, value)
    
    model.fit(s, x, t, jnp.array([0, 1]))
    score, value = model.infer(s, t)
    print(score, value)