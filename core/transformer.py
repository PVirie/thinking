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



class TransformerDecoderBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, enc_out, x, train):
    
        # Masked multi-head self-attention
        mask = nn.attention.make_causal_mask(x[:, :, 0])
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False
        )(x, mask=mask)
        
        # Skip connection and layer norm
        x = x + attn_output
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Multi-head cross-attention, only attend to the same slot to fuse target state
        eye_mask = jnp.eye(x.shape[1], dtype=jnp.float32)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.dropout_rate
        )(inputs_q=x, inputs_k=enc_out, inputs_v=enc_out, mask=eye_mask)
        
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
        # decoder_input has shape [batch, seq_len, d_model]
        # encoder_input has shape [batch, seq_len, d_model]

        d_model = encoder_input.shape[-1] if self.d_model == -1 else self.d_model

        # Potentially project input to d_model size
        if d_model != encoder_input.shape[-1]:
            encoder_input = nn.Dense(d_model)(encoder_input)
            decoder_input = nn.Dense(d_model)(decoder_input)

        # Stack of Transformer blocks
        for num_heads, d_ff in self.layers:
            decoder_input = TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=self.dropout_rate
            )(encoder_input, decoder_input, train)

        # Final layer norm and projection to output dimension
        decoder_input = nn.LayerNorm(epsilon=1e-6)(decoder_input)
        decoder_output = nn.Dense(self.output_dim)(decoder_input)

        return decoder_output


class Value_Score_Module(nn.Module):
    slots: int
    output_dim: int
    stacked_transformer: StackedTransformer

    @nn.compact
    def __call__(self, s, x, t):
        logits = self.stacked_transformer(t, s, True)
        # flatten
        seq_len = logits.shape[1]
        x = jnp.reshape(x, (-1, self.output_dim))
        logits = jnp.reshape(logits, (-1, self.slots))
        
        keys = jax.nn.softmax(logits, axis=1)
        Vs = nn.Dense(self.slots*self.output_dim)(keys)
        Ss = nn.Dense(self.slots)(keys)

        Vs = jnp.reshape(Vs, (-1, self.slots, self.output_dim))

        Vl = jnp.expand_dims(x, axis=2)
        denom = jnp.linalg.norm(Vs, axis=2, keepdims=True) * jnp.linalg.norm(Vl, axis=1, keepdims=True)
        dot_scores = jnp.linalg.matmul(Vs, Vl) / denom

        # dot_scores has shape [batch, memory_size, 1]
        dot_scores = jnp.reshape(dot_scores, (-1, self.slots))

        max_indices = jnp.argmax(dot_scores, axis=1, keepdims=True)
        s = jnp.take_along_axis(Ss, max_indices, axis=1)
        v = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
        
        v = jnp.reshape(v, (-1, seq_len, self.output_dim))
        s = jnp.reshape(s, (-1, seq_len))

        return v, s


class Value_Score_Module_Test(Value_Score_Module):


    @nn.compact
    def __call__(self, s, t):
        logits = self.stacked_transformer(t, s, False)
        # take only last context
        logits = logits[:, -1, :]

        keys = jax.nn.softmax(logits, axis=1)
        Vs = nn.Dense(self.slots*self.output_dim)(keys)
        Ss = nn.Dense(self.slots)(keys)

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


@partial(jax.jit, static_argnames=['context_length'])
def pad_sequence(s, t, scores, masks, context_length):
    # pad s, t with zeros
    sp = jnp.pad(s[:, :-1, :], ((0, 0), (context_length, 0), (0, 0)), mode='constant', constant_values=0)
    tp = jnp.pad(t[:, :-1, :], ((0, 0), (context_length, 0), (0, 0)), mode='constant', constant_values=0)

    # x = s rolled backward by 1
    xp = jnp.pad(s[:, 1:, :], ((0, 0), (context_length, 0), (0, 0)), mode='constant', constant_values=0)

    # pad score and mask
    scorep = jnp.pad(scores[:, :-1], ((0, 0), (context_length, 0)), mode='constant', constant_values=0)
    maskp = jnp.pad(masks[:, :-1], ((0, 0), (context_length, 0)), mode='constant', constant_values=0)

    return sp, xp, tp, scorep, maskp


class Model(base.Model):

    def __init__(self, input_dims, context_length, hidden_size, layers, memory_size, lr=1e-4):
        super().__init__("model", "transformer")

        rng = jax.random.key(42)
        self.rng, self.dropout_rng = jax.random.split(rng)
        self.input_dims = input_dims
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.layers = layers

        stack_transformer = StackedTransformer(layers=layers, d_model=input_dims, output_dim=hidden_size)
        self.train_model = Value_Score_Module(slots=memory_size, output_dim=input_dims, stacked_transformer=stack_transformer)
        self.test_model = Value_Score_Module_Test(slots=memory_size, output_dim=input_dims, stacked_transformer=stack_transformer)

        variables = self.train_model.init(
            {'params': self.rng, 'dropout': self.dropout_rng}, 
            jnp.empty((1, self.context_length, input_dims), jnp.float32), 
            jnp.empty((1, self.context_length, input_dims), jnp.float32),
            jnp.empty((1, self.context_length, input_dims), jnp.float32),
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


    def fit_sequence(self, s, t, scores, masks=None, context=None):
        # s has shape (N, seq_len, dim), t has shape (N, seq_len, dim), scores has shape (N, seq_len), masks has shape (N, seq_len)
        # seq_len = learning_length + context_length - 1

        if masks is None:
            masks = jnp.ones(scores.shape)

        sp, xp, tp, scorep, maskp = pad_sequence(s, t, scores, masks, self.context_length)

        self.state, loss = train_step(self.state, sp, xp, tp, scorep, maskp, self.state.dropout_key)

        self.is_updated = True
        return loss


    def infer(self, s, t, context=None):
        # s has shape (N, context_length, dim), t has shape (N, dim)
        
        s = jnp.reshape(s, (-1, self.context_length, self.input_dims))
        t = jnp.reshape(t, (-1, 1, self.input_dims))
        t = jnp.repeat(t, self.context_length, axis=1)

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
    model = Model(4, 1, 4, [(4, 4), (4, 4)], 4, 0.001)

    eye = jnp.eye(4, dtype=jnp.float32)
    S = jnp.array([[eye[0, :], eye[1, :], eye[2, :], eye[3, :]]])
    T = jnp.array([[eye[3, :], eye[3, :], eye[3, :], eye[3, :]]])

    for i in range(1000):
        loss = model.fit_sequence(S, T, jnp.array([[0.72, 0.81, 0.9, 1.0]]))

    s = jnp.array([[eye[0, :]], [eye[1, :]], [eye[2, :]], [eye[3, :]]])
    t = jnp.array([eye[3, :], eye[3, :], eye[3, :], eye[3, :]])

    value, score = model.infer(s, t)
    print("Loss:", loss)
    print("Score:", score)
    print("Value:", value)