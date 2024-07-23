import jax.numpy as jnp
import os

from . import base

class Model(base.Model):

    def __init__(self, dims):
        self.class_name = "table"
        self.input_dims = dims
        # make [[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0] ...]]
        eye = jnp.eye(dims, dtype=jnp.float32)
        eye_1 = jnp.expand_dims(eye, axis=0)
        eye_1 = jnp.tile(eye_1, (dims, 1, 1))
        eye_2 = jnp.expand_dims(eye, axis=1)
        eye_2 = jnp.tile(eye_2, (1, dims, 1))
        self.key = jnp.concatenate([eye_2, eye_1], axis=-1)
        self.key = jnp.reshape(self.key, (-1, dims * 2))

        # self.key = jnp.reshape(features, (-1, self.input_dims * 2))
        self.score = jnp.zeros([dims * dims, 1], jnp.float32)
        self.value = jnp.zeros([dims * dims, dims], jnp.float32)


    def get_class_parameters(self):
        return {"class_name": self.class_name, "input_dims": self.input_dims}


    def fit(self, s, x, t, scores, masks=1.0):
        # s has shape (N, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))
        x = jnp.reshape(x, (-1, self.input_dims))

        queries = jnp.concatenate([s, t], axis=-1)
        batch = jnp.reshape(queries, (-1, self.input_dims * 2))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        # logits has shape [N, len(key)]
        # find max key for each batch
        argmax_logits = jnp.argmax(logits, axis=1)

        # update indices where scores are higher than current scores
        update_indices = (scores > self.score[argmax_logits]) * masks
        score_updates = (1 - update_indices) * self.score[argmax_logits] + update_indices * scores
        # update score at argmax_logits with updates
        self.score = self.score.at[argmax_logits].set(score_updates)

        # only select values that are argmax_logits
        value_updates = (1 - update_indices) * self.value[argmax_logits] + update_indices * x
        # update values at argmax_logits with value_updates
        self.value = self.value.at[argmax_logits].set(value_updates)

        return jnp.mean(score_updates)


    def infer(self, s, t):
        # s has shape (N, dim), t has shape (N, dim)

        # for simple model only use the last state
        queries = jnp.concatenate([s, t], axis=-1)
        batch = jnp.reshape(queries, (-1, self.input_dims * 2))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        # logits has shape [N, len(key)]
        # find max key for each batch
        argmax_logits = jnp.argmax(logits, axis=1)
        # return best score, value

        best_score = jnp.reshape(self.score[argmax_logits], [queries.shape[0]])
        best_value = jnp.reshape(self.value[argmax_logits], [queries.shape[0], -1])

        return best_value, best_score


    def save(self, path):
        jnp.save(os.path.join(path, "score.npy"), self.score)
        jnp.save(os.path.join(path, "value.npy"), self.value)


    @staticmethod
    def load(path, class_parameters):
        model = Model(class_parameters["input_dims"])
        model.score = jnp.load(os.path.join(path, "score.npy"))
        model.value = jnp.load(os.path.join(path, "value.npy"))
        return model



if __name__ == "__main__":
    model = Model(4)
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