import os

from typing import Sequence, Union, List, Any

import jax
import jax.random
import jax.numpy as jnp
import flax.linen as nn
from clu import metrics
import flax
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax  

from flax import serialization

try:
    from . import base
except:
    import base



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


class TransformerDecoderBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    training: bool = False

    @nn.compact
    def __call__(self, enc_out, x):
        # Masked multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not self.training,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False
        )(x, mask=False)
        # no mask, this is not sequence prediction, we already feed step-by-step query
        
        # Skip connection and layer norm
        x = x + attn_output
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Multi-head cross-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not self.training,
            dropout_rate=self.dropout_rate
        )(inputs_q=x, inputs_k=enc_out, inputs_v=enc_out)
        
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
    def __call__(self, encoder_input, decoder_input):

        d_model = encoder_input.shape[-1] if self.d_model == -1 else self.d_model

        # Potentially project input to d_model size
        if d_model != encoder_input.shape[-1]:
            encoder_input = nn.Dense(d_model)(encoder_input)
            decoder_input = nn.Dense(d_model)(decoder_input)

        # Stack of Transformer blocks
        for num_heads, d_ff in self.layers:
            encoder_output = TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=self.dropout_rate,
                training=self.training
            )(encoder_input)
            decoder_input = TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=self.dropout_rate,
                training=self.training
            )(encoder_output, decoder_input)

        # Final layer norm and projection to output dimension
        decoder_input = nn.LayerNorm(epsilon=1e-6)(decoder_input)
        decoder_output = nn.Dense(self.output_dim)(decoder_input)

        return decoder_output

    

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: Metrics
  dropout_key: jax.Array



def loss_fn(params, encoder_input, decoder_input, labels, masks, state, dropout_train_key):
    logits = state.apply_fn(
        {'params': params}, 
        encoder_input,
        decoder_input, 
        rngs={'dropout': dropout_train_key})
    loss = jnp.mean(((logits - labels) ** 2) * masks)
    return loss

jitted_loss = jax.jit(jax.value_and_grad(loss_fn, argnums=(0)))

@jax.jit
def train_step(state, encoder_input, decoder_input, labels, masks, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    loss, grads = jitted_loss(state.params, encoder_input, decoder_input, labels, masks, state, dropout_train_key)
    return state.apply_gradients(grads=grads), loss


class Model(base.Model):

    def __init__(self, layers, input_dims, lr=1e-4):
        super().__init__("model", "transformer")

        rng = jax.random.key(42)
        self.rng, self.dropout_rng = jax.random.split(rng)
        self.input_dims = input_dims
        self.layers = layers
        self.model = StackedTransformer(layers=layers, output_dim=input_dims, training=True)
        self.predict_model = StackedTransformer(layers=layers, output_dim=input_dims, training=False)

        variables = self.model.init(
            {'params': self.rng, 'dropout': self.dropout_rng}, 
            jnp.empty((1, input_dims), jnp.float32), 
            jnp.empty((1, input_dims), jnp.float32)
        )

        self.learning_rate = lr
        self.momentum = 0.9

        self.state = TrainState.create(
            apply_fn=self.model.apply, 
            params=variables["params"], 
            tx=optax.adamw(self.learning_rate),
            metrics=Metrics.empty(), 
            dropout_key=self.dropout_rng
        )


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "input_dims": self.input_dims,
            "layers": self.layers,
            "lr": self.learning_rate,
        }


    def save(self, path):
        bytes_output = serialization.to_bytes(self.state)
        with open(os.path.join(path, "state.bin"), 'wb') as f:
            f.write(bytes_output)
        # logging.info(serialization.to_state_dict(self.state))
        self.is_updated = False

    def load(self, path):
        with open(os.path.join(path, "state.bin"), 'rb') as f:
            bytes_output = f.read()
        self.state = serialization.from_bytes(self.state, bytes_output)
        # logging.info(serialization.to_state_dict(self.state))


    def fit(self, s, x, t, scores, masks, context=None):
        # s has shape (N, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)

        s = jnp.reshape(s, (-1, self.input_dims))
        t = jnp.reshape(t, (-1, self.input_dims))
        x = jnp.reshape(x, (-1, self.input_dims))

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))

        self.state, loss = train_step(self.state, t, s, x, masks, self.state.dropout_key)

        self.is_updated = True
        return loss


    def infer(self, s, t, context=None):
        # s has shape (N, dim), t has shape (N, dim)
        # t will be fed to encoder input while s will be fed to decoder input
        
        s = jnp.reshape(s, (-1, self.input_dims))
        t = jnp.reshape(t, (-1, self.input_dims))

        logits = self.predict_model.apply(
            {
                'params': self.state.params,
            },
            t, 
            s,
            mutable=False, 
            rngs={
                'dropout': self.state.dropout_key
            }
        )

        unflatten = jnp.reshape(logits, [-1, self.input_dims])
        return unflatten, 0



if __name__ == "__main__":
    model = Model([(4, 4), (4, 4)], 4, 0.01)

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