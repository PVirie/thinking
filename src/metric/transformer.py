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


class MultiHeadSelfAttention(nn.Module):
    num_heads: int
    d_model: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        d_head = self.d_model // self.num_heads
        
        # Project input to query, key, and value vectors
        qkv = nn.Dense(self.d_model * 3)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Split heads
        q = q.reshape(x.shape[0], -1, self.num_heads, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(x.shape[0], -1, self.num_heads, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(x.shape[0], -1, self.num_heads, d_head).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        attn_logits = attn_logits / jnp.sqrt(d_head)
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -jnp.inf)
        attn_weights = nn.softmax(attn_logits)

        # Apply dropout to attention weights if not in a deterministic context
        attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=deterministic)

        # Attention output
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(x.shape[0], -1, self.d_model)
        
        # Final dense layer to produce output
        output = nn.Dense(self.d_model)(attn_output)
        return output


class TransformerEncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    training: bool = False

    @nn.compact
    def __call__(self, x):
        # Multi-head self-attention
        attn_output = nn.SelfAttention(
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
        rng = jax.random.PRNGKey(42)
        self.rng, self.dropout_rng = jax.random.split(rng)
        self.input_dims = input_dims
        self.model = StackedTransformer(layers=[(16, 1024), (16, 1024), (16, 1024)], output_dim=1, training=True)
        self.predict_model = StackedTransformer(layers=[(16, 1024), (16, 1024), (16, 1024)], output_dim=1, training=False)

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
    
    model = StackedTransformer(layers=[(10, 10), (10, 10)], output_dim=2)
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