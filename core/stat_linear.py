"""
Statistic with linear kernel for computing observation probabilities.
Author: P.Virie

"""

import jax
import jax.random
import jax.numpy as jnp
import os
import pickle
from functools import partial
import optax

try:
    from . import base, linear
except:
    import base, linear


@jax.jit
def compute_stats(Q, params):
    K = params[0]
    S = params[1]
    indices = jax.nn.softmax(jnp.matmul(Q, jnp.transpose(K)), axis=-1)

    P_ = jnp.matmul(indices, S)
    return P_


@jax.jit
def compute_error(Q, params):
    K = params[0]
    S = params[1]

    indices = jax.nn.softmax(jnp.matmul(Q, jnp.transpose(K)), axis=-1)
    Q_ = jnp.matmul(indices, K)
    P_ = jnp.matmul(indices, S)
    return jnp.mean((Q - Q_) ** 2) + jnp.mean((1 - P_) ** 2) + jnp.mean(S**2) * 0.1


value_grad_function = jax.jit(jax.value_and_grad(compute_error, argnums=(1)))


# loop training jit
@partial(jax.jit, static_argnames=['optimizer'])
def train_step(optimizer, params, r_key, opt_state, query):
    r_key, subkey = jax.random.split(r_key)
    loss, grads = value_grad_function(query, params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, r_key, opt_state


class Model(base.Stat_Model):

    def __init__(self, input_dims, hidden_size, lr=0.01, r_seed=42):
        super().__init__("stat", "head")

        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.r_seed = r_seed

        r_key =  jax.random.key(r_seed)
        r_key, subkey = jax.random.split(r_key)
        key = jax.random.normal(subkey, (hidden_size, input_dims)) * 0.1

        stats = jnp.ones([self.hidden_size, 1], jnp.float32) / self.hidden_size

        self.r_key = r_key
        self.params = (key, stats)

        self.lr = lr

        self.optimizer = optax.adamw(self.lr)
        self.opt_state = self.optimizer.init(self.params)


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "input_dims": self.input_dims,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "r_seed": self.r_seed
        }


    def save(self, path):
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump(self.params, f)
        self.is_updated = False


    def load(self, path):
        with open(os.path.join(path, "params.pkl"), "rb") as f:
            self.params = pickle.load(f)
        self.is_updated = False


    def accumulate(self, s):
        # s has shape (N, dim)
        query = jnp.reshape(s, (-1, self.input_dims))

        loss, self.params, self.r_key, self.opt_state  = train_step(self.optimizer, self.params, self.r_key, self.opt_state, query)

        self.is_updated = True
        return loss


    def infer(self, s):
        # s has shape (N, dim)
        query = jnp.reshape(s, (-1, self.input_dims))
        # get only the first half of the linear core's key
        stats = compute_stats(query, self.params)
        # return shape (N)
        return jnp.reshape(stats, (-1))
    



if __name__ == "__main__":

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :]])
    t = jnp.array([eye[3, :], eye[3, :]])

    model = Model(4, 8, 0.1)
    print(model.infer(eye))

    for i in range(100):
        model.accumulate(s)
    print(model.infer(eye))
