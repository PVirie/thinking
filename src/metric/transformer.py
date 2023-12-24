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


class TransformerEncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    training: bool = False

    @nn.compact
    def __call__(self, x):
        # Multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not self.training,
            dropout_rate=self.dropout_rate
        )(x)
        
        # Skip connection and layer norm
        x = x + attn_output
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Position-wise feed-forward network
        ff_output = nn.Dense(self.d_ff)(x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate)(ff_output, deterministic=not self.training)
        ff_output = nn.Dense(self.d_model)(ff_output)
        
        # Second skip connection and layer norm
        x = x + ff_output
        x = nn.LayerNorm(epsilon=1e-6)(x)

        return x


class StackedTransformer(nn.Module):
    layers: Sequence[tuple]  # Each tuple is (num_heads, d_ff)
    output_dim: int
    d_model: int = -1  # Will infer from input if not provided
    dropout_rate: float = 0.1
    training: bool = False

    @nn.compact
    def __call__(self, x):
        d_model = x.shape[-1] if self.d_model == -1 else self.d_model

        # Potentially project input to d_model size
        if d_model != x.shape[-1]:
            x = nn.Dense(d_model)(x)

        # Stack of Transformer blocks
        for num_heads, d_ff in self.layers:
            x = TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=self.dropout_rate,
                training=self.training
            )(x)

        # Final layer to project to output dimension
        x = nn.Dense(self.output_dim)(x)

        return x

    

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: Metrics
  dropout_key: jax.Array



@jax.jit
def train_step(state, batch, labels, masks, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch, rngs={'dropout': dropout_train_key})
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
        rng = jax.random.PRNGKey(42)
        self.rng, self.dropout_rng = jax.random.split(rng)
        self.input_dims = input_dims
        self.model = StackedTransformer(layers=[(16, 1024), (16, 1024), (16, 1024)], output_dim=1, training=True)
        self.predict_model = StackedTransformer(layers=[(16, 1024), (16, 1024), (16, 1024)], output_dim=1, training=False)

        learning_rate = 1e-4
        momentum = 0.9
        variables = self.model.init({'params': self.rng, 'dropout': self.dropout_rng}, jnp.empty((1, input_dims * 2), jnp.float32))
        tx = optax.sgd(learning_rate, momentum)

        self.state = TrainState.create(
            apply_fn=self.model.apply, 
            params=variables["params"], 
            tx=tx,
            metrics=Metrics.empty(), 
            dropout_key=self.dropout_rng
        )

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

        self.state, loss = train_step(self.state, batch, labels, masks, self.state.dropout_key)

        return loss
        

    def likelihood(self, s, t, cartesian=False):
        features = metric_base.make_features(s, t, cartesian)
        
        batch = jnp.reshape(features, (-1, self.input_dims * 2))

        logits = self.predict_model.apply(
            {
                'params': self.state.params,
            }, 
            batch, 
            mutable=False, 
            rngs={
                'dropout': self.state.dropout_key
            }
        )

        # reshape logits back to features shape except the last dimension
        unflatten = jnp.reshape(logits, features.shape[:-1])
        
        return unflatten
    
# To do: added metric computation and plot https://flax.readthedocs.io/en/latest/getting_started.html


if __name__ == '__main__':
    
    model = StackedTransformer(layers=[(10, 10), (10, 10)], output_dim=2)
    key = jax.random.PRNGKey(0)
    batch = jax.random.normal(key, (32, 10))
    variables = model.init({'params': key, 'dropout': key}, batch)
    output = model.apply(variables, batch)
    print(output)

    net = Model(16)
    s = [Node(np.random.rand(16)), Node(np.random.rand(16))]
    s2 = [Node(np.random.rand(16)), Node(np.random.rand(16))]
    t = Node(np.ones(16))
    distance = net.likelihood(s, t)
    distance2 = net.likelihood(s2, t)
    print("Before", distance, distance2)
    for i in range(1000):
        loss = net.learn(s, t, 1.0, jnp.array([1.0, 1.0]))
        loss2 = net.learn(s2, t, 0.5, jnp.array([1.0, 1.0]))
        if i % 100 == 0:
            print(loss, loss2)
    distance = net.likelihood(s, t)
    distance2 = net.likelihood(s2, t)
    print("After", distance, distance2)
    net.save("weights")
    net.load("weights")
    distance = net.likelihood(s, t)
    distance2 = net.likelihood(s2, t)
    print("After load", distance, distance2)