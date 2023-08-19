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
    training: bool = False

    @nn.compact
    def __call__(self, x):
        # x has shape [batch, dim]
        y = nn.Dense(features=self.features)(x)
        y = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(y)
        y = nn.relu(y)
        y = nn.Dense(features=self.features)(y)
        y = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(y)
        y = nn.relu(y)
        return x + y

class Resnet(nn.Module):
    # not convolutional, just dense
    # layer is [num_features, num_features, ...]
    layers: Sequence[int]
    output_dim: int
    input_dim: int = -1 # -1 means it will be inferred from the input
    training: bool = False
    
    @nn.compact
    def __call__(self, x):
        # x has shape [batch, dim]
        last_dims = self.input_dim
        for features in self.layers:
            # when last last_dims != features use dense
            if last_dims != features:
                x = nn.Dense(features)(x)
                x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(x)
                x = nn.relu(x)
            x = ResnetBlock(features)(x)
            last_dims = features
        x = nn.Dense(features=self.output_dim)(x)
        return x
    

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  batch_stats: flax.core.FrozenDict[str, Any]
  metrics: Metrics



@jax.jit
def train_step(state, batch, labels, masks):

    def loss_fn(params):
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch, mutable=['batch_stats'])
        loss = jnp.mean(mse_loss(logits, labels)*masks)
        return loss, new_model_state
    
    (loss, new_model_state), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)
    
    return state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
    ), loss


@jax.vmap
def mse_loss(logit, label):
    return (logit - label) ** 2


def deep_get_data(x):
    # if x is a node, return data
    if isinstance(x, Node):
        return x.data
    elif isinstance(x, list):
        return [deep_get_data(y) for y in x]


class Model(metric_base.Model):

    def __init__(self, input_dims):
        self.rng = jax.random.PRNGKey(42)
        self.input_dims = input_dims
        self.model = Resnet(layers=[16, 8, 4], output_dim=1, training=True)
        self.predict_model = Resnet(layers=[16, 8, 4], output_dim=1, training=False)

        learning_rate = 1e-4
        momentum = 0.9
        variables = self.model.init(self.rng, jnp.ones((1, input_dims), jnp.float32))
        tx = optax.sgd(learning_rate, momentum)

        self.state = TrainState.create(
            apply_fn=self.model.apply, params=variables["params"], tx=tx,
            metrics=Metrics.empty(), batch_stats=variables["batch_stats"])

    def save(self, path):
        bytes_output = serialization.to_bytes(self.state)
        with open(os.path.join(path, "resnet.bin"), 'wb') as f:
            f.write(bytes_output)
        # print(serialization.to_state_dict(self.state))


    def load(self, path):
        with open(os.path.join(path, "resnet.bin"), 'rb') as f:
            bytes_output = f.read()
        serialization.from_bytes(self.state, bytes_output)
        # print(serialization.to_state_dict(self.state))


    def learn(self, s, t, labels, masks):
        s = jnp.array(deep_get_data(s))
        t = jnp.array(deep_get_data(t))

        # To do: now we use simple t - start as the feature, we can use more complex features
        features = t - s

        batch = jnp.reshape(features, (-1, self.input_dims))
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
        

    def distance(self, s, t, to_numpy=True):
        s = jnp.array(deep_get_data(s))
        t = jnp.array(deep_get_data(t))
        
        features = t - s
        batch = jnp.reshape(features, (-1, self.input_dims))

        logits = self.predict_model.apply({'params': self.state.params, 'batch_stats': self.state.batch_stats}, batch, mutable=False)

        # reshape logits back to features shape except the last dimension
        unflatten = jnp.reshape(logits, features.shape[:-1])
        
        if to_numpy:
            return unflatten
        else:
            return unflatten
    
# To do: added metric computation and plot https://flax.readthedocs.io/en/latest/getting_started.html


if __name__ == '__main__':
    
    model = Resnet(layers=[8, 4], output_dim=2)
    key = jax.random.PRNGKey(0)
    batch = jax.random.normal(key, (32, 10))
    variables = model.init(key, batch)
    output = model.apply(variables, batch)
    print(output)

    resnet = Model(16)
    s = [Node(np.random.rand(16)), Node(np.random.rand(16))]
    t = Node(np.ones(16))
    labels = 1.0
    resnet.learn(s, t, labels, jnp.array([1.0, 2.0]))
    distance = resnet.distance(s, t)
    print(distance)