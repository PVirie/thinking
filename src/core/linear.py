import jax
import jax.random
import jax.numpy as jnp
import os
from functools import partial

try:
    from . import base
except:
    import base


@partial(jax.jit, static_argnames=['dim_size', 'memory_size', 'batch_size'])
def compute_value_score(Q, K, Wv, Ws, dim_size, memory_size, batch_size):
    logit = jax.nn.softmax(jnp.matmul(Q, jnp.transpose(K)), axis=-1)
    Ss = jnp.matmul(logit, Ws)
    Vs = jnp.matmul(logit, Wv)
    Vs = jnp.reshape(Vs, (batch_size, memory_size, dim_size))

    # S has shape [batch, memory_size]
    max_indices = jnp.argmax(Ss, axis=1, keepdims=True)
    s = jnp.take_along_axis(Ss, max_indices, axis=1)
    v = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
    return v, s


def compute_error(Q, S, V, M, K, Wv, Ws, dim_size, memory_size, batch_size):
    logit = jax.nn.softmax(jnp.matmul(Q, jnp.transpose(K)), axis=-1)
    Ss = jnp.matmul(logit, Ws)
    Vs = jnp.matmul(logit, Wv)
    Vs = jnp.reshape(Vs, (batch_size, memory_size, dim_size))
    
    # V has shape [batch, dim_size]
    # Vs has shape [batch, memory_size, dim_size]
    # find the best match index in the memory using cosine similarity

    Vl = jnp.expand_dims(V, axis=2)
    denom = jnp.linalg.norm(Vs, axis=2, keepdims=True) * jnp.linalg.norm(Vl, axis=1, keepdims=True)
    dot_scores = jnp.linalg.matmul(Vs, Vl) / denom

    max_indices = jnp.argmax(dot_scores, axis=1, keepdims=True)
    S_ = jnp.take_along_axis(jnp.expand_dims(Ss, axis=-1), max_indices, axis=1)
    V_ = jnp.take_along_axis(Vs, max_indices, axis=1)

    error_S = M * (S - S_)**2
    error_V = M * (V - V_)**2
    return jnp.mean(error_V) + jnp.mean(error_S)


value_grad_function = jax.jit(jax.value_and_grad(compute_error, argnums=(4, 5, 6)), static_argnames=['dim_size', 'memory_size', 'batch_size'])

# loop training jit
@partial(jax.jit, static_argnames=['epoch_size', 'dim_size', 'memory_size', 'batch_size'])
def loop_training(Q, V, S, M, K, Wv, Ws, iteration, lr, epoch_size, dim_size, memory_size, batch_size):

    temperature = jnp.exp(-iteration/2000)
    for i in range(epoch_size):
        loss, (g_K, g_Wv, g_Ws) = value_grad_function(Q, S, V, M, K, Wv, Ws, dim_size, memory_size, batch_size)
        K = K - lr*g_K
        Wv = Wv - lr*g_Wv
        Ws = Ws - lr*g_Ws
    return K, Wv, Ws, loss


class Model(base.Model):

    def __init__(self, hidden_size, input_dims, lr=0.01, epoch_size=1, iteration=0):
        super().__init__("model", "linear")

        self.hidden_size = hidden_size
        self.input_dims = input_dims
        self.memory_size = 16

        r_key = jax.random.key(42)
        r_key, subkey = jax.random.split(r_key)
        self.key = jax.random.normal(subkey, (hidden_size, input_dims * 2)) * 0.01
        
        r_key, subkey = jax.random.split(r_key)
        self.score = jax.random.normal(subkey, [hidden_size, self.memory_size * 1]) * 0.01
        r_key, subkey = jax.random.split(r_key)
        self.value = jax.random.normal(subkey, [hidden_size, self.memory_size * input_dims]) * 0.01

        self.lr = lr
        self.epoch_size = epoch_size
        self.iteration = iteration


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "hidden_size": self.hidden_size,
            "input_dims": self.input_dims, 
            "lr": self.lr,
            "epoch_size": self.epoch_size,
            "iteration": self.iteration
        }


    def save(self, path):
        jnp.save(os.path.join(path, "key.npy"), self.key)
        jnp.save(os.path.join(path, "score.npy"), self.score)
        jnp.save(os.path.join(path, "value.npy"), self.value)

        self.is_updated = False

    def load(self, path):
        self.key = jnp.load(os.path.join(path, "key.npy"))
        self.score = jnp.load(os.path.join(path, "score.npy"))
        self.value = jnp.load(os.path.join(path, "value.npy"))


    def fit(self, s, x, t, scores, masks=1.0):
        # s has shape (N, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))
        x = jnp.reshape(x, (-1, self.input_dims))

        batch = jnp.concatenate([s, t], axis=-1)
        query = jnp.reshape(batch, (-1, self.input_dims * 2))

        batch_size = query.shape[0]
        self.key, self.value, self.score, loss = loop_training(query, x, scores, masks, self.key, self.value, self.score, self.iteration, self.lr, self.epoch_size, self.input_dims, self.memory_size, batch_size)
        self.iteration += 1

        self.is_updated = True
        return loss


    def infer(self, s, t):
        # s has shape (N, dim), t has shape (N, dim)

        # for simple model only use the last state
        batch = jnp.concatenate([s, t], axis=-1)
        query = jnp.reshape(batch, (-1, self.input_dims * 2))

        batch_size = query.shape[0]
        best_value, best_score = compute_value_score(query, self.key, self.value, self.score, self.input_dims, self.memory_size, batch_size)

        return best_value, best_score


if __name__ == "__main__":
    model = Model(8, 4, 0.1, 100, iteration=0)

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :]])
    t = jnp.array([eye[3, :], eye[3, :]])

    loss = model.fit(s, x, t, jnp.array([0.9, 0.5]))
    value, score = model.infer(s, t)
    print("Loss:", loss)
    print("Score:", score)
    print("Value:", value)
    
    loss = model.fit(s, x, t, jnp.array([0.5, 0.6]))
    value, score = model.infer(s, t)
    print("Loss:", loss)
    print("Score:", score)
    print("Value:", value)