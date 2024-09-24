"""
Linear kernel for storing and fetching versioned values
Author: P.Virie

This is a simple asymmetric linear model implementing with JAX.
"""

import jax
import jax.random
import jax.numpy as jnp
from jax import lax
import os
import pickle
from functools import partial
import optax

try:
    from . import base
except:
    import base


@partial(jax.jit, static_argnames=['window_size', 'batch_size', 'sequence_length', 'dim'])
def unfold(x, window_size, batch_size, sequence_length, dim):
  """
  Unfolds an input tensor into sliding windows.

  Args:
    x: Input tensor of shape [batch, sequence, dim].
    window_size: Size of the sliding window.

  Returns:
    Unfolded tensor of shape [batch, sequence - window_size + 1, window_size, dim].
  """

  def body_fun(i, result):
      window = lax.dynamic_slice(x, [0, i, 0], [batch_size, window_size, dim])
      return result.at[i].set(window)

  result = jnp.zeros([sequence_length - window_size + 1, batch_size, window_size, dim])
  result = lax.fori_loop(0, sequence_length - window_size + 1, body_fun, result)
  return result.transpose([1, 0, 2, 3])  # Rearrange to [batch, sequence - window_size + 1, window_size, dim]


@jax.jit
def query(Q, params):
    K = params[0]
    Wv_0 = params[1]
    Ws_0 = params[2]
    
    # Q has shape [batch, dim * (context_length + 1)]
    # K has shape [hidden_size, dim * (context_length + 1)]
    
    logit = jnp.matmul(Q, jnp.transpose(K)) / jnp.sqrt(K.shape[1])
    weights = jax.nn.softmax(logit, axis=-1)

    Vs = jnp.matmul(weights, Wv_0)
    Ss = jnp.matmul(weights, Ws_0)

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

    S_ = jnp.take_along_axis(Ss, max_indices, axis=1)
    V_ = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
    V_ = jnp.reshape(V_, (batch_size, dim_size))

    error_S = M * (S - S_)**2
    error_V = M * (V - V_)**2

    # suppress other slot score to 0
    error_C = jnp.mean(Ss ** 2)
    return jnp.mean(error_V) + jnp.mean(error_S) + error_C * 0.1


value_grad_function = jax.jit(jax.value_and_grad(compute_error, argnums=(4)), static_argnames=['dim_size', 'memory_size', 'batch_size'])

@partial(jax.jit, static_argnames=['context_length', 'input_dims'])
def make_query(s, t, context_length, input_dims):
    batch = jnp.concatenate([jnp.reshape(s, (-1, input_dims * context_length)), t], axis=-1)
    query = jnp.reshape(batch, (-1, input_dims * (context_length + 1)))
    return query


@partial(jax.jit, static_argnames=['optimizer', 'input_dims', 'memory_size', 'batch_size'])
def train_step(optimizer, params, r_key, opt_state, query, x, scores, masks, input_dims, memory_size, batch_size):
    r_key, subkey = jax.random.split(r_key)
    loss, grads = value_grad_function(query, x, scores, masks, params, subkey, input_dims, memory_size, batch_size)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, r_key, opt_state



class Model(base.Model):

    def __init__(self, input_dims, context_length, hidden_size, memory_size=16, lr=0.01, iteration=0, r_seed=42):
        super().__init__("model", "linear")

        self.input_dims = input_dims
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.r_seed = r_seed

        r_key = jax.random.key(r_seed)
        r_key, subkey = jax.random.split(r_key)
        key = jax.random.normal(subkey, (hidden_size, input_dims * (context_length + 1))) * 0.1
        
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
            "input_dims": self.input_dims, 
            "context_length": self.context_length,
            "hidden_size": self.hidden_size,
            "memory_size": self.memory_size,
            "lr": self.lr,
            "iteration": self.iteration,
            "r_seed": self.r_seed
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


    def fit(self, s, x, t, scores, masks=None, context=None):
        # s has shape (N, context_length, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)

        if masks is None:
            masks = jnp.ones(scores.shape)

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))
        x = jnp.reshape(x, (-1, self.input_dims))

        query = make_query(s, t, self.context_length, self.input_dims)
        batch_size = query.shape[0]

        loss, self.params, self.r_key, self.opt_state = train_step(self.optimizer, self.params, self.r_key, self.opt_state, query, x, scores, masks, self.input_dims, self.memory_size, batch_size)

        self.iteration += 1

        self.is_updated = True
        return loss


    def fit_sequence(self, s, x, t, scores, masks=None, context=None):
        # s has shape (N, seq_len, dim), x has shape (N, seq_len, dim), t has shape (N, seq_len, dim), scores has shape (N, seq_len), masks has shape (N, seq_len)
        # seq_len = learning_length + context_length - 1

        if masks is None:
            masks = jnp.ones(scores.shape)

        # first pad with zeros
        sp = jnp.pad(s, ((0, 0), (self.context_length - 1, 0), (0, 0)), mode='constant', constant_values=0)

        # then unroll by shift and tile
        # unrolled_s has shape (N, seq_len, context_length, dim)
        unrolled_s = unfold(sp, self.context_length, *sp.shape)
                            
        return self.fit(
            jnp.reshape(unrolled_s, (-1, self.context_length, self.input_dims)),
            jnp.reshape(x, (-1, self.input_dims)), 
            jnp.reshape(t, (-1, self.input_dims)), 
            jnp.reshape(scores, (-1)), 
            jnp.reshape(masks, (-1)), context)


    def infer(self, s, t, context=None):
        # s has shape (N, context_length, dim), t has shape (N, dim)

        if s.shape[1] < self.context_length:
            # pad input
            s = jnp.pad(s, ((0, 0), (0, self.context_length - s.shape[1]), (0, 0)), mode='constant', constant_values=0)
        elif s.shape[1] > self.context_length:
            s = s[:, -self.context_length:, :]

        query = make_query(s, t, self.context_length, self.input_dims)
        batch_size = query.shape[0]

        best_value, best_score = compute_value_score(query, self.params, self.input_dims, self.memory_size, batch_size)

        best_value = jnp.reshape(best_value, [batch_size, -1])
        best_score = jnp.reshape(best_score, [batch_size])

        return best_value, best_score


if __name__ == "__main__":
    from datetime import datetime
    # get number of milliseconds since midnight of January 1, 1970
    millis = datetime.now().microsecond
    model = Model(4, 2, 8, 4, 0.01, iteration=0, r_seed=millis)

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
