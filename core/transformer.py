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
from functools import partial

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

    @nn.compact
    def __call__(self, x, train):
        # Multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.dropout_rate
        )(x)
        
        # Skip connection and layer norm
        x = x + attn_output
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Position-wise feed-forward network
        ff_output = nn.Dense(self.d_ff)(x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate)(ff_output, deterministic=not train)
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

    @nn.compact
    def __call__(self, enc_out, x, train):
        # Masked multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
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
            deterministic=not train,
            dropout_rate=self.dropout_rate
        )(inputs_q=x, inputs_k=enc_out, inputs_v=enc_out)
        
        # Skip connection and layer norm
        x = x + attn_output
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Position-wise feed-forward network
        ff_output = nn.Dense(self.d_ff)(x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate)(ff_output, deterministic=not train)
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

    @nn.compact
    def __call__(self, encoder_input, decoder_input, train):

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
                dropout_rate=self.dropout_rate
            )(encoder_input, train)
            decoder_input = TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=self.dropout_rate
            )(encoder_output, decoder_input, train)

        # Final layer norm and projection to output dimension
        decoder_input = nn.LayerNorm(epsilon=1e-6)(decoder_input)
        decoder_output = nn.Dense(self.output_dim)(decoder_input)

        return decoder_output


class Value_Score_Module(nn.Module):
    slots: int
    output_dim: int
    stacked_transformer: StackedTransformer


    def setup(self):
        self.value_pathway = nn.Dense(self.slots*self.output_dim)
        self.score_pathway = nn.Dense(self.slots)


    def __call__(self, s, x, t, scores):
        logits = self.stacked_transformer(t, s, True)
        keys = jax.nn.softmax(logits, axis=1)
        Vs = self.value_pathway(keys)
        Ss = self.score_pathway(keys)

        Vs = jnp.reshape(Vs, (-1, self.slots, self.output_dim))

        Vl = jnp.expand_dims(x, axis=2)
        denom = jnp.linalg.norm(Vs, axis=2, keepdims=True) * jnp.linalg.norm(Vl, axis=1, keepdims=True)
        dot_scores = jnp.linalg.matmul(Vs, Vl) / denom
        max_indices = jnp.argmax(dot_scores, axis=1)

        s = jnp.take_along_axis(Ss, max_indices, axis=1)
        v = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
        v = jnp.reshape(v, (-1, self.output_dim))

        return v, s


class Value_Score_Module_Test(Value_Score_Module):
    def __call__(self, s, t):
        logits = self.stacked_transformer(t, s, False)
        keys = jax.nn.softmax(logits, axis=1)
        Vs = self.value_pathway(keys)
        Ss = self.score_pathway(keys)

        Vs = jnp.reshape(Vs, (-1, self.slots, self.output_dim))

        # Ss has shape [batch, memory_size]
        max_indices = jnp.argmax(Ss, axis=1, keepdims=True)
        s = jnp.take_along_axis(Ss, max_indices, axis=1)
        v = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
        v = jnp.reshape(v, (-1, self.output_dim))

        return v, s

    

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: Metrics
  dropout_key: jax.Array



def loss_fn(params, s, x, t, scores, masks, state, dropout_train_key):
    v_, scores_ = state.apply_fn(
        {'params': params}, 
        s,
        x, 
        t,
        scores,
        rngs={'dropout': dropout_train_key})
    
    error_S = masks * (scores - scores_)**2
    error_V = masks * (x - v_)**2

    # suppress other slot score to 0
    error_C = jnp.mean(scores_ ** 2)
    return jnp.mean(error_V) + jnp.mean(error_S) + error_C * 0.1

jitted_loss = jax.jit(jax.value_and_grad(loss_fn, argnums=(0)))

@jax.jit
def train_step(state, s, x, t, scores, masks, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    loss, grads = jitted_loss(state.params, s, x, t, scores, masks, state, dropout_train_key)
    return state.apply_gradients(grads=grads), loss


class Model(base.Model):

    def __init__(self, hidden_size, input_dims, memory_size, layers, lr=1e-4):
        super().__init__("model", "transformer")

        rng = jax.random.key(42)
        self.rng, self.dropout_rng = jax.random.split(rng)
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.layers = layers

        stack_transformer = StackedTransformer(layers=layers, output_dim=hidden_size)
        self.train_model = Value_Score_Module(slots=memory_size, output_dim=input_dims, stacked_transformer=stack_transformer)
        self.test_model = Value_Score_Module_Test(slots=memory_size, output_dim=input_dims, stacked_transformer=stack_transformer)

        variables = self.train_model.init(
            {'params': self.rng, 'dropout': self.dropout_rng}, 
            jnp.empty((1, input_dims), jnp.float32), 
            jnp.empty((1, input_dims), jnp.float32),
            jnp.empty((1, input_dims), jnp.float32),
            jnp.empty((1), jnp.float32),
        )

        self.learning_rate = lr
        self.momentum = 0.9

        self.state = TrainState.create(
            apply_fn=self.train_model.apply, 
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
        x = jnp.reshape(x, (-1, self.input_dims))
        t = jnp.reshape(t, (-1, self.input_dims))

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))

        self.state, loss = train_step(self.state, s, x, t, scores, masks, self.state.dropout_key)

        self.is_updated = True
        return loss


    def infer(self, s, t, context=None):
        # s has shape (N, dim), t has shape (N, dim)
        
        s = jnp.reshape(s, (-1, self.input_dims))
        t = jnp.reshape(t, (-1, self.input_dims))

        best_value, best_score = self.test_model.apply(
            {
                'params': self.state.params,
            },
            s, 
            t,
            mutable=False, 
            rngs={
                'dropout': self.state.dropout_key
            }
        )

        best_value = jnp.reshape(best_value, [-1, self.input_dims])
        best_score = jnp.reshape(best_score, [-1])
        return best_value, best_score



if __name__ == "__main__":
    model = Model(16, 4, 4, [(4, 4), (4, 4)], 0.001)

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