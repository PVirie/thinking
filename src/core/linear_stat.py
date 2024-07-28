import jax
import jax.random
import jax.numpy as jnp
import os

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


def compute_gradient(Q, K, S, lr):
    logit = compute_logit(Q, K)
    sum_logit = jnp.mean(logit, axis=0, keepdims=True)
    return S * (1-lr) + lr * jnp.transpose(sum_logit)

# extremely faster with jit
update_function = jax.jit(compute_gradient)

class Model(base.Stat_Model):

    def __init__(self, linear_core: linear.Model, lr=0.01, epoch_size=100):
        super().__init__("stat", "linear")

        self.hidden_size = linear_core.hidden_size
        self.input_dims = linear_core.input_dims
        self.linear_core = linear_core

        self.stats = jnp.ones([self.hidden_size, 1], jnp.float32) / self.hidden_size

        self.lr = lr
        self.epoch_size = epoch_size


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "linear_core": self.linear_core.instance_id,
            "lr": self.lr,
            "epoch_size": self.epoch_size
        }


    def save(self, path):
        jnp.save(os.path.join(path, "stats.npy"), self.stats)
        self.is_updated = False


    def load(self, path):
        self.stats = jnp.load(os.path.join(path, "stats.npy"))


    def accumulate(self, s):
        # s has shape (N, dim)
        query = jnp.reshape(s, (-1, self.input_dims))
        for i in range(self.epoch_size):
            self.stats = update_function(query, self.linear_core.key[:, :self.input_dims], self.stats, self.lr)

        self.is_updated = True

    def infer(self, s):
        # s has shape (N, dim)
        query = jnp.reshape(s, (-1, self.input_dims))
        # get only the first half of the linear core's key
        stats = compute_stats(query, self.linear_core.key[:, :self.input_dims], self.stats)
        # return shape (N)
        return jnp.reshape(stats, (-1))
    



if __name__ == "__main__":
    linear_core_model = linear.Model(8, 4, 0.1, 100)

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :]])
    t = jnp.array([eye[3, :], eye[3, :]])

    loss = linear_core_model.fit(s, x, t, jnp.array([1, 1]))

    model = Model(linear_core_model, 0.1, 100)
    print(model.infer(eye))
    model.accumulate(eye)
    print(model.infer(eye))
