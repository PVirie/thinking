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
def compute_stats(Q, K, S):
    logit = jnp.matmul(Q, jnp.transpose(K))
    normed_logit = jnp.linalg.norm(logit, axis=1, keepdims=True)
    normed_S = jnp.linalg.norm(S, axis=0, keepdims=True)
    return jnp.abs(jnp.matmul(logit, S) / (normed_logit * normed_S))


@jax.jit
def compute_error(Q, K, S):
    logit = jnp.matmul(Q, jnp.transpose(K))
    sum_logit = jnp.transpose(jnp.mean(logit, axis=0, keepdims=True))
    return jnp.mean((S - sum_logit) ** 2)


value_grad_function = jax.jit(jax.value_and_grad(compute_error, argnums=(2)))


# loop training jit
@partial(jax.jit, static_argnames=['optimizer'])
def train_step(optimizer, params, r_key, opt_state, query, key):
    r_key, subkey = jax.random.split(r_key)
    loss, grads = value_grad_function(query, key, params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, r_key, opt_state


class Model(base.Stat_Model):

    def __init__(self, linear_core: linear.Model, lr=0.01):
        super().__init__("stat", "head")

        self.hidden_size = linear_core.hidden_size
        self.input_dims = linear_core.input_dims
        self.linear_core = linear_core

        r_key = jax.random.key(42)
        stats = jnp.ones([self.hidden_size, 1], jnp.float32) / self.hidden_size

        self.r_key = r_key
        self.params = stats

        self.lr = lr

        self.optimizer = optax.adamw(self.lr)
        self.opt_state = self.optimizer.init(self.params)


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "linear_core": self.linear_core.instance_id,
            "lr": self.lr
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

        loss, self.params, self.r_key, self.opt_state  = train_step(self.optimizer, self.params, self.r_key, self.opt_state, query, self.linear_core.params[0][:, :self.input_dims])

        self.is_updated = True
        return loss


    def infer(self, s):
        # s has shape (N, dim)
        query = jnp.reshape(s, (-1, self.input_dims))
        # get only the first half of the linear core's key
        stats = compute_stats(query, self.linear_core.params[0][:, :self.input_dims], self.params)
        # return shape (N)
        return jnp.reshape(stats, (-1))
    

if __name__ == "__main__":
    linear_core_model = linear.Model(8, 4, 4, 0.01)

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :]])
    t = jnp.array([eye[3, :], eye[3, :]])

    for i in range(1000):
        loss = linear_core_model.fit(s, x, t, jnp.array([1, 1]), 1.0)

    model = Model(linear_core_model, 0.1)
    print(model.infer(eye))

    for i in range(1000):
        model.accumulate(eye)
    
    print(model.infer(eye))
