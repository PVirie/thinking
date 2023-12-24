from node import Node, Node_tensor_2D
import jax.numpy as jnp
class Model:

    def __init__(self, dims):
        pass

    def learn(self, s, t, labels, masks):
        pass

    def distance(self, s, t):
        pass


def deep_get_data(x):
    # if x is a node, return data
    if isinstance(x, Node):
        return x.data
    elif isinstance(x, list):
        return [deep_get_data(y) for y in x]


def make_features(s, t, cartesian=False):
    s = jnp.array(deep_get_data(s))
    t = jnp.array(deep_get_data(t))

    if cartesian:
        # expand dim to s (s.shape[0], t.shape[0], dim) and t (s.shape[0], t.shape[0], dim)
        if len(s.shape) == 2:
            s_ = jnp.expand_dims(s, axis=1)
            s_ = jnp.tile(s_, (1, t.shape[0], 1))

        if len(t.shape) == 2:
            t_ = jnp.expand_dims(t, axis=0)
            t_ = jnp.tile(t_, (s.shape[0], 1, 1))

        # the output has shape (s.shape[0], t.shape[0], dim * 2)
        features = jnp.concatenate([t_, s_], axis=-1)
    else:
        # broad cast t to s shape
        if len(s.shape) > len(t.shape):
            t = jnp.expand_dims(t, axis=0)
            t = jnp.tile(t, (s.shape[0], 1))
        features = jnp.concatenate([t, s], axis=-1)
    
    return features