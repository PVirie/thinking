import os

import metric_base
from node import Node, Node_tensor_2D
from typing import Sequence, Union, List, Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from clu import metrics
import flax
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax  

from flax import serialization


class ResnetBlock(nn.Module):
    # not convolutional, just dense
    features: int

    @nn.compact
    def __call__(self, x):
        # x has shape [batch, dim]
        y = nn.Dense(features=self.features)(x)
        y = nn.relu(y)
        y = nn.Dense(features=self.features)(y)
        y = nn.relu(y)
        return x + y

class Resnet(nn.Module):
    # not convolutional, just dense
    # layer is [num_features, num_features, ...]
    layers: Sequence[int]
    output_dim: int
    input_dim: int = -1 # -1 means it will be inferred from the input
    
    @nn.compact
    def __call__(self, x):
        # x has shape [batch, dim]
        last_dims = self.input_dim
        for features in self.layers:
            # when last last_dims != features use dense
            if last_dims != features:
                x = nn.Dense(features)(x)
                x = nn.relu(x)
            x = ResnetBlock(features)(x)
            last_dims = features
        x = nn.Dense(features=self.output_dim)(x)
        return nn.sigmoid(x)
    

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: Metrics


@jax.jit
def train_step(state, batch, labels, masks):

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch)
        loss = jnp.mean(mse_loss(logits, labels)*masks)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    return state.apply_gradients(
        grads=grads,
    ), loss


@jax.vmap
def mse_loss(logit, label):
    return (logit - label) ** 2


class Model(metric_base.Model):

    def __init__(self, input_dims):
        self.rng = jax.random.PRNGKey(42)
        self.input_dims = input_dims
        self.model = Resnet(layers=[8, 8], output_dim=1)

        learning_rate = 1e-4
        momentum = 0.9
        variables = self.model.init(self.rng, jnp.empty((1, input_dims * 2), jnp.float32))
        tx = optax.adam(learning_rate, momentum)

        self.state = TrainState.create(
            apply_fn=self.model.apply, 
            params=variables["params"], 
            tx=tx,
            metrics=Metrics.empty()
        )

    def save(self, path):
        bytes_output = serialization.to_bytes(self.state)
        with open(os.path.join(path, "resnet.bin"), 'wb') as f:
            f.write(bytes_output)
        # print(serialization.to_state_dict(self.state))


    def load(self, path):
        with open(os.path.join(path, "resnet.bin"), 'rb') as f:
            bytes_output = f.read()
        self.state = serialization.from_bytes(self.state, bytes_output)
        # print(serialization.to_state_dict(self.state))



    def learn(self, s, t, labels, masks, cartesian=False):

        features = metric_base.make_features(s, t, cartesian)

        batch = jnp.reshape(features, (-1, self.input_dims * 2))
        # if labels is a float, we need to reshape it to (batch, 1)
        if isinstance(labels, float):
            labels = jnp.ones((batch.shape[0], 1)) * labels
        else:
            labels = jnp.reshape(labels, (-1, 1))
            
        # if masks is a float, we need to reshape it to (batch, 1)
        if isinstance(masks, float):
            masks = jnp.ones((batch.shape[0], 1)) * masks
        else:
            masks = jnp.reshape(masks, (-1, 1))

        self.state, loss = train_step(self.state, batch, labels, masks)

        return loss
        

    def likelihood(self, s, t, cartesian=False):

        features = metric_base.make_features(s, t, cartesian)
        
        batch = jnp.reshape(features, (-1, self.input_dims * 2))

        logits = self.model.apply({'params': self.state.params}, batch, mutable=False)

        # reshape logits back to features shape except the last dimension
        unflatten = jnp.reshape(logits, features.shape[:-1])
        
        return unflatten
    
# To do: added metric computation and plot https://flax.readthedocs.io/en/latest/getting_started.html


if __name__ == '__main__':
    
    model = Resnet(layers=[8, 4], output_dim=2)
    key = jax.random.PRNGKey(0)
    batch = jax.random.normal(key, (32, 10))
    variables = model.init(key, batch)
    output = model.apply(variables, batch)
    print(output)

    s = [Node(np.random.rand(16)), Node(np.random.rand(16))]
    s2 = [Node(np.random.rand(16)), Node(np.random.rand(16))]
    t = Node(np.ones(16))

    resnet = Model(16)
    distance = resnet.likelihood(s, t)
    distance2 = resnet.likelihood(s2, t)
    print("Before", distance, distance2)
    for i in range(1000):
        loss = resnet.learn(s, t, 1.0, jnp.array([1.0, 1.0]))
        loss2 = resnet.learn(s2, t, 0.5, jnp.array([1.0, 1.0]))
        if i % 100 == 0:
            print(loss, loss2)
    distance = resnet.likelihood(s, t)
    distance2 = resnet.likelihood(s2, t)
    print("After", distance, distance2)
    resnet.save("weights")

    resnet2 = Model(16)
    resnet2.load("weights")
    distance = resnet2.likelihood(s, t)
    distance2 = resnet2.likelihood(s2, t)
    print("After load", distance, distance2)