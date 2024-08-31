import jax
import jax.random
import jax.numpy as jnp
import os
import pickle
from functools import partial
import optax
from optax import contrib

try:
    from . import base
except:
    import base


@jax.jit
def query(Q, params):
    K = params[0]
    Wv_0 = params[1]
    Ws_0 = params[2]
    
    # Q has shape [batch, dim * 2]
    # K has shape [hidden_size, dim * 2]
    
    logit = jnp.matmul(Q, jnp.transpose(K)) / jnp.sqrt(K.shape[1])
    weights = jax.nn.softmax(logit, axis=-1)

    Vs = jnp.matmul(weights, Wv_0)
    Ss = jnp.matmul(weights, Ws_0)

    # logit is square diff
    # logit = jnp.sum((jnp.expand_dims(Q, axis=1) - jnp.expand_dims(K, axis=0))**2, axis=2)
    # min_indices = jnp.argmin(logit, axis=1, keepdims=True)
    # L = jnp.take_along_axis(logit, min_indices, axis=1)
    # Vs = jnp.take_along_axis(Wv, min_indices, axis=0) * L
    # Ss = jnp.take_along_axis(Ws, min_indices, axis=0) * L

    return Vs, Ss


@partial(jax.jit, static_argnames=['dim_size', 'memory_size', 'batch_size'])
def compute_value_score(Q, params, dim_size, memory_size, batch_size):
    Vs, Ss = query(Q, params)
    Vs = jnp.reshape(Vs, (batch_size, memory_size, dim_size))

    # S has shape [batch, memory_size]
    max_indices = jnp.argmax(Ss, axis=1, keepdims=True)
    s = jnp.take_along_axis(Ss, max_indices, axis=1)
    v = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
    v = jnp.reshape(v, (batch_size, dim_size))
    return v, s


def compute_error(Q, V, S, M, params, r_key, dim_size, memory_size, batch_size):
    Vs, Ss = query(Q, params)
    Vs = jnp.reshape(Vs, (batch_size, memory_size, dim_size))
    
    # V has shape [batch, dim_size]
    # Vs has shape [batch, memory_size, dim_size]
    # find the best match index in the memory using cosine similarity

    Vl = jnp.expand_dims(V, axis=2)
    denom = jnp.linalg.norm(Vs, axis=2, keepdims=True) * jnp.linalg.norm(Vl, axis=1, keepdims=True)
    dot_scores = jnp.linalg.matmul(Vs, Vl) / denom
    max_indices = jnp.argmax(dot_scores, axis=1)

    # S__ = jnp.take_along_axis(Ss, max_indices, axis=1)
    # random_indices = jax.random.randint(r_key, (batch_size, 1), 0, memory_size)
    # max_indices = jnp.where(S__ < 0.01, random_indices, max_indices)

    S_ = jnp.take_along_axis(Ss, max_indices, axis=1)
    V_ = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
    V_ = jnp.reshape(V_, (batch_size, dim_size))

    error_S = M * (S - S_)**2
    error_V = M * (V - V_)**2

    # suppress other slot score to 0
    error_C = jnp.mean(Ss ** 2)
    return jnp.mean(error_V) + jnp.mean(error_S) + error_C * 0.1


value_grad_function = jax.jit(jax.value_and_grad(compute_error, argnums=(4)), static_argnames=['dim_size', 'memory_size', 'batch_size'])

# # loop training jit
# @partial(jax.jit, static_argnames=['dim_size', 'memory_size', 'batch_size'])
# def loop_training(Q, V, S, M, K, Wv, Ws, iteration, lr, r_key, dim_size, memory_size, batch_size):
#     temperature = jnp.exp(-iteration/2000)
#     r_key, subkey = jax.random.split(r_key)
#     loss, (g_K, g_Wv, g_Ws) = value_grad_function(Q, V, S, M, K, Wv, Ws, subkey, dim_size, memory_size, batch_size)
#     K = K - lr*g_K
#     Wv = Wv - lr*g_Wv
#     Ws = Ws - lr*g_Ws
#     return K, Wv, Ws, loss


@partial(jax.jit, static_argnames=['input_dims'])
def make_query(s, t, input_dims):
    batch = jnp.concatenate([s, t], axis=-1)
    query = jnp.reshape(batch, (-1, input_dims * 2))
    return query


@partial(jax.jit, static_argnames=['optimizer', 'input_dims', 'memory_size', 'batch_size'])
def train_step(optimizer, params, r_key, opt_state, query, x, scores, masks, input_dims, memory_size, batch_size):
    r_key, subkey = jax.random.split(r_key)
    loss, grads = value_grad_function(query, x, scores, masks, params, subkey, input_dims, memory_size, batch_size)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, r_key, opt_state



class Model(base.Model):

    def __init__(self, hidden_size, input_dims, memory_size=16, lr=0.01, iteration=0):
        super().__init__("model", "linear")

        self.hidden_size = hidden_size
        self.input_dims = input_dims
        self.memory_size = memory_size

        r_key = jax.random.key(42)
        r_key, subkey = jax.random.split(r_key)
        key = jax.random.normal(subkey, (hidden_size, input_dims * 2)) * 0.1
        
        r_key, subkey = jax.random.split(r_key)
        value_0 = jax.random.normal(subkey, [hidden_size, self.memory_size * self.input_dims]) * 0.1

        r_key, subkey = jax.random.split(r_key)
        score_0 = jax.random.normal(subkey, [hidden_size, self.memory_size]) * 0.01

        self.params = (key, value_0, score_0)

        self.r_key = r_key

        self.lr = lr
        self.iteration = iteration

        self.optimizer = optax.adamw(self.lr)
        self.opt_state = self.optimizer.init(self.params)


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "hidden_size": self.hidden_size,
            "input_dims": self.input_dims, 
            "lr": self.lr,
            "iteration": self.iteration
        }


    def save(self, path):
        # save params
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump(self.params, f)

        # save optimizer state
        with open(os.path.join(path, "opt_state.pkl"), "wb") as f:
            pickle.dump(self.opt_state, f)

        # save iteration
        with open(os.path.join(path, "iteration.pkl"), "wb") as f:
            pickle.dump(self.iteration, f)

        self.is_updated = False

    def load(self, path):
        # load params
        with open(os.path.join(path, "params.pkl"), "rb") as f:
            self.params = pickle.load(f)

        # load optimizer state
        with open(os.path.join(path, "opt_state.pkl"), "rb") as f:
            self.opt_state = pickle.load(f)

        # load iteration
        with open(os.path.join(path, "iteration.pkl"), "rb") as f:
            self.iteration = pickle

        self.is_updated = False


    def fit(self, s, x, t, scores, masks, context=None):
        # s has shape (N, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))
        x = jnp.reshape(x, (-1, self.input_dims))

        query = make_query(s, t, self.input_dims)
        batch_size = query.shape[0]

        # self.key, self.value, self.score, loss = loop_training(query, x, scores, masks, self.key, self.value, self.score, self.iteration, self.lr, self.r_key, self.input_dims, self.memory_size, batch_size)
        
        loss, self.params, self.r_key, self.opt_state = train_step(self.optimizer, self.params, self.r_key, self.opt_state, query, x, scores, masks, self.input_dims, self.memory_size, batch_size)

        self.iteration += 1

        self.is_updated = True
        return loss


    def infer(self, s, t, context=None):
        # s has shape (N, dim), t has shape (N, dim)

        query = make_query(s, t, self.input_dims)
        batch_size = query.shape[0]

        best_value, best_score = compute_value_score(query, self.params, self.input_dims, self.memory_size, batch_size)

        best_value = jnp.reshape(best_value, [batch_size, -1])
        best_score = jnp.reshape(best_score, [batch_size])

        return best_value, best_score


if __name__ == "__main__":
    model = Model(16, 4, 4, 0.01, iteration=0)

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :], eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :], eye[2, :], eye[0, :]])
    t = jnp.array([eye[3, :], eye[3, :], eye[3, :], eye[3, :]])

    for i in range(1000):
        loss = model.fit(s, x, t, jnp.array([0.9, 0.9, 0.5, 0.5]), 1.0)

    value, score = model.infer(s, t)
    print("Loss:", loss)
    print("Score:", score)
    print("Value:", value)
    