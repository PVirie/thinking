"""
Statistic table model for one-hot graph only (for model checking)
Author: P.Virie

"""

import jax.numpy as jnp
import os

try:
    from . import base
except:
    import base


class Model(base.Model):

    def __init__(self, dims):
        super().__init__("stat", "table")

        self.input_dims = dims
        # make [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.key = jnp.eye(dims, dtype=jnp.float32)

        self.stat = jnp.zeros([dims, 1], jnp.float32)


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "dims": self.input_dims
        }


    def save(self, path):
        jnp.save(os.path.join(path, "stat.npy"), self.stat)

        self.is_updated = False

    def load(self, path):
        self.stat = jnp.load(os.path.join(path, "stat.npy"))
    

    def accumulate(self, s):
        query = jnp.reshape(s, (-1, self.input_dims))
        logits = jnp.matmul(query, jnp.transpose(self.key))
        argmax_logits = jnp.argmax(logits, axis=1)
        self.stat = self.stat.at[argmax_logits].add(1)
        return 0.0


    def infer(self, s):
        query = jnp.reshape(s, (-1, self.input_dims))
        logits = jnp.matmul(query, jnp.transpose(self.key))
        argmax_logits = jnp.argmax(logits, axis=1)
        stats = jnp.reshape(self.stat[argmax_logits], [query.shape[0]])
        return stats





if __name__ == "__main__":

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :]])
    t = jnp.array([eye[3, :], eye[3, :]])

    model = Model(4)
    print(model.infer(eye))
    model.accumulate(s)
    print(model.infer(eye))
    model.accumulate(eye)
    print(model.infer(eye))
