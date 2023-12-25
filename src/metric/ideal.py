import metric_base
from node import Node, Node_tensor_2D
import jax.numpy as jnp
import os

class Model:

    def __init__(self, dims):
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
        self.value = jnp.zeros([dims * dims], jnp.float32)

    def learn(self, s, t, labels, masks, cartesian=False):
        features = metric_base.make_features(s, t, cartesian)
        batch = jnp.reshape(features, (-1, self.input_dims * 2))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        # logits has shape [batch, key]
        # find max key for each batch
        argmax_logits = jnp.argmax(logits, axis=1)

        # if masks is a float, we need to reshape it to (batch, 1)
        if isinstance(masks, float):
            masks = jnp.ones((batch.shape[0])) * masks
        else:
            masks = jnp.reshape(masks, (-1))

        labels = jnp.reshape(labels, (-1))
        updates = jnp.where(labels * masks > self.value[argmax_logits], labels, self.value[argmax_logits])
        # update value at argmax_logits with updates
        self.value = self.value.at[argmax_logits].set(updates)

        return jnp.mean(updates)

    def likelihood(self, s, t, cartesian=False):
        features = metric_base.make_features(s, t, cartesian)
        batch = jnp.reshape(features, (-1, self.input_dims * 2))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        # logits has shape [batch, key]
        # find max key for each batch
        argmax_logits = jnp.argmax(logits, axis=1)
        # access value
        return jnp.reshape(self.value[argmax_logits], features.shape[:-1])

    def save(self, path):
        jnp.save(os.path.join(path, "ideal.npy"), self.value)

    def load(self, path):
        self.value = jnp.load(os.path.join(path, "ideal.npy"))




if __name__ == "__main__":
    model = Model(3)
    print(model.key)

    eye = jnp.eye(3, dtype=jnp.float32)
    eye_nodes = [Node(x) for x in eye]
    print(eye_nodes)

    model.learn(eye_nodes, eye_nodes, 0.5, jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0]), cartesian=True)
    value = model.likelihood(eye_nodes, eye_nodes, cartesian=True)
    print(value)
    
    model.learn(eye_nodes, eye_nodes, 1.0, jnp.array([1, 0, 0, 1, 1, 0, 0, 0, 0]), cartesian=True)
    value = model.likelihood(eye_nodes, eye_nodes, cartesian=True)
    print(value)