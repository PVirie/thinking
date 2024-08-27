import jax
import jax.random
import jax.numpy as jnp
import os
from functools import partial

try:
    from . import base, linear
except:
    import base, linear


def compute_logit(Q, K):
    return jnp.matmul(Q, jnp.transpose(K))


def compute_stats(Q, K, S):
    logit = compute_logit(Q, K)
    normed_logit = jnp.linalg.norm(logit, axis=1, keepdims=True)
    normed_S = jnp.linalg.norm(S, axis=0, keepdims=True)
    return jnp.abs(jnp.matmul(logit, S) / (normed_logit * normed_S))


def compute_error(Q, K, S):
    # To do: implement the error function

    return 0


# extremely faster with jit
value_grad_function = jax.jit(jax.value_and_grad(compute_error, argnums=(1, 2)))


# loop training jit

@partial(jax.jit, static_argnames=['lr', 'epoch_size'])
def loop_training(Q, K, S, lr, epoch_size):
    for i in range(epoch_size):
        loss, (g_K, g_S) = value_grad_function(Q, K, S)
        K = K - lr*g_K
        S = S - lr*g_S
    return K, S, loss

class Model(base.Stat_Model):

    def __init__(self, hidden_size, input_dims, lr=0.01, epoch_size=100):
        super().__init__("stat", "linear")

        self.hidden_size = hidden_size
        self.input_dims = input_dims

        r_key = jax.random.key(42)
        r_key, subkey = jax.random.split(r_key)
        self.key = jax.random.normal(subkey, (hidden_size, input_dims * 2))*0.01
        self.stats = jnp.ones([self.hidden_size, 1], jnp.float32) / self.hidden_size

        self.lr = lr
        self.epoch_size = epoch_size


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "hidden_size": self.hidden_size,
            "input_dims": self.input_dims, 
            "lr": self.lr,
            "epoch_size": self.epoch_size
        }


    def save(self, path):
        jnp.save(os.path.join(path, "key.npy"), self.key)
        jnp.save(os.path.join(path, "stats.npy"), self.stats)
        self.is_updated = False


    def load(self, path):
        self.key = jnp.load(os.path.join(path, "key.npy"))
        self.stats = jnp.load(os.path.join(path, "stats.npy"))


    def accumulate(self, s):
        # s has shape (N, dim)
        query = jnp.reshape(s, (-1, self.input_dims))

        self.key, self.stats, loss = loop_training(query, self.key, self.stats, self.lr, self.epoch_size)

        self.is_updated = True


    def infer(self, s):
        # s has shape (N, dim)
        query = jnp.reshape(s, (-1, self.input_dims))
        # get only the first half of the linear core's key
        stats = compute_stats(query, self.key, self.stats)
        # return shape (N)
        return jnp.reshape(stats, (-1))
    



if __name__ == "__main__":

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :]])
    t = jnp.array([eye[3, :], eye[3, :]])

    model = Model(8, 4, 0.1, 100)
    print(model.infer(eye))
    model.accumulate(eye)
    print(model.infer(eye))
