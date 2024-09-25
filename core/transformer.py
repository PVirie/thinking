import os
from typing import Sequence, Union, List, Any
from collections.abc import Callable
import jax
import jax.random
import jax.numpy as jnp
import flax.linen as nn
from flax import core, struct              # Flax dataclasses
import optax
import pickle
from functools import partial

try:
    from . import base
except:
    import base


def create_n_step_causal_mask(batch_size, seq_len, n):
    # unlike usual causal mask, this mask will have n steps backward only
    # for example n = 2
    # [1, 0, 0, 0]
    # [1, 1, 0, 0]
    # [0, 1, 1, 0]
    # [0, 0, 1, 1]

    mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)
    for i in range(seq_len):
        start_idx = max(0, i - n + 1)
        mask = mask.at[i, start_idx:i+1].set(1)

    # Convert the mask to shape (batch_size, 1, seq_len, seq_len) for broadcasting
    mask = jnp.expand_dims(mask, axis=0)  # Shape: (1, seq_len, seq_len)
    mask = jnp.expand_dims(mask, axis=0)  # Shape: (batch_size, 1, seq_len, seq_len)
    mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))

    return mask


def create_self_attention_mask(batch_size, sequence_length):
    mask = jnp.eye(sequence_length)
    mask = mask[jnp.newaxis, :, :]
    mask = jnp.tile(mask, (batch_size, 1, 1))
    mask = mask[:, jnp.newaxis, :, :]
    return mask


class TransformerDecoderBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, enc_in, x, mask, train):
    
        batch = x.shape[0]
        seq_len = x.shape[1]

        # Masked multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False
        )(inputs_q=x, inputs_k=x, inputs_v=x, mask=mask)
        
        # Skip connection and layer norm
        x = x + attn_output
        x = nn.LayerNorm(epsilon=1e-6)(x)
        
        # Multi-head cross-attention, only attend to the same slot to fuse target state
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.d_model, 
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=not train,
            dropout_rate=self.dropout_rate
        )(inputs_q=x, inputs_k=enc_in, inputs_v=enc_in, mask=create_self_attention_mask(batch, seq_len))
        
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
    context_length: int
    d_model: int = -1  # Will infer from input if not provided
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, encoder_input, decoder_input, train):
        # decoder_input has shape [batch, seq_len, d_model]
        # encoder_input has shape [batch, seq_len, d_model]

        d_model = encoder_input.shape[-1] if self.d_model == -1 else self.d_model

        # Project input to d_model size
        encoder_input = nn.Dense(d_model)(encoder_input)
        decoder_input = nn.Dense(d_model)(decoder_input)

        batch = decoder_input.shape[0]
        seq_len = decoder_input.shape[1]
        mask = create_n_step_causal_mask(batch, seq_len, self.context_length)

        # Stack of Transformer blocks
        for num_heads, d_ff in self.layers:
            decoder_input = TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=self.dropout_rate
            )(encoder_input, decoder_input, mask, train)

        # Final layer norm and projection to output dimension
        decoder_input = nn.LayerNorm(epsilon=1e-6)(decoder_input)
        decoder_output = nn.Dense(self.output_dim)(decoder_input)

        return decoder_output


class Value_Score_Module(nn.Module):
    records: int
    slots: int
    output_dim: int
    query_backbone: StackedTransformer
    value_score_backbone: StackedTransformer

    @nn.compact
    def __call__(self, s, x, t):

        seq_len = s.shape[1]

        # Query

        logits = self.query_backbone(t, s, True)
        keys = jax.nn.softmax(logits, axis=-1)

        Wvs = self.value_score_backbone(t, s, True)
        Wv = Wvs[:, :, :self.records * self.slots * self.output_dim]
        Ws = Wvs[:, :, self.records * self.slots * self.output_dim:]

        keys = jnp.reshape(keys, (-1, self.records, 1))
        Vs = jnp.matmul(jnp.reshape(Wv, (-1, self.slots * self.output_dim, self.records)), keys)
        Ss = jnp.matmul(jnp.reshape(Ws, (-1, self.slots, self.records)), keys)
        
        # Vs = nn.Dense(self.slots * self.output_dim)(keys)
        # Ss = nn.Dense(self.slots)(keys)

        Vs = jnp.reshape(Vs, (-1, self.slots, self.output_dim))
        Ss = jnp.reshape(Ss, (-1, self.slots))

        # Access

        x = jnp.reshape(x, (-1, self.output_dim))
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

        return v, s, Ss


class Value_Score_Module_Test(Value_Score_Module):
    @nn.compact
    def __call__(self, s, t):
        # Query

        logits = self.query_backbone(t, s, False)
        keys = jax.nn.softmax(logits, axis=-1)
        keys = keys[:, -1, :]

        Wvs = self.value_score_backbone(t, s, False)
        Wv = Wvs[:, -1, :self.records * self.slots * self.output_dim]
        Ws = Wvs[:, -1, self.records * self.slots * self.output_dim:]

        keys = jnp.reshape(keys, (-1, self.records, 1))
        Vs = jnp.matmul(jnp.reshape(Wv, (-1, self.slots * self.output_dim, self.records)), keys)
        Ss = jnp.matmul(jnp.reshape(Ws, (-1, self.slots, self.records)), keys)

        # Vs = nn.Dense(self.slots * self.output_dim)(keys)
        # Ss = nn.Dense(self.slots)(keys)
        
        Vs = jnp.reshape(Vs, (-1, self.slots, self.output_dim))
        Ss = jnp.reshape(Ss, (-1, self.slots))

        # Access

        max_indices = jnp.argmax(Ss, axis=1, keepdims=True)
        s = jnp.take_along_axis(Ss, max_indices, axis=1)
        v = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
        
        v = jnp.reshape(v, (-1, self.output_dim))

        return v, s

    
class Train_state(struct.PyTreeNode):
    model_fn: Callable = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    dropout_key: jax.Array
    step: int | jax.Array


    def get_rng(self):
        return jax.random.fold_in(key=self.dropout_key, data=self.step)


    def apply_gradients(self, grads):
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state
        )


    def save(self, path):
        # save params
        with open(os.path.join(path, "model_state.pkl"), "wb") as f:
            pickle.dump(self.params, f)
        # save opt_state
        with open(os.path.join(path, "opt_state.pkl"), "wb") as f:
            pickle.dump(self.opt_state, f)
        # save dropout_key
        with open(os.path.join(path, "dropout_key.pkl"), "wb") as f:
            pickle.dump(self.dropout_key, f)
        # save step
        with open(os.path.join(path, "step.pkl"), "wb") as f:
            pickle.dump(self.step, f)


    def load(self, path):
        # load params
        with open(os.path.join(path, "model_state.pkl"), "rb") as f:
            params = pickle.load(f)
        # load opt_state
        with open(os.path.join(path, "opt_state.pkl"), "rb") as f:
            opt_state = pickle.load(f)
        # load dropout_key
        with open(os.path.join(path, "dropout_key.pkl"), "rb") as f:
            dropout_key = pickle.load(f)
        # load step
        with open(os.path.join(path, "step.pkl"), "rb") as f:
            step = pickle.load(f)
        return self.replace(
            params=params,
            opt_state=opt_state,
            dropout_key=dropout_key,
            step=step
        )


def loss_fn(params, model_fn, s, x, t, scores, masks, dropout_train_key):
    v_, scores_, Ss = model_fn({'params': params}, s, x, t, rngs={'dropout': dropout_train_key})
    
    error_S = jnp.sum(masks * (scores - scores_)**2) / jnp.sum(masks)
    error_V = jnp.sum(masks * jnp.mean((x - v_)**2, axis=-1)) / jnp.sum(masks)
    # suppress other slot score to 0
    error_C = jnp.mean(Ss ** 2)

    return error_V + error_S + error_C * 0.1


jitted_loss = jax.jit(jax.value_and_grad(loss_fn, argnums=(0)), static_argnames=['model_fn'])


@partial(jax.jit, static_argnames=['context_length', 'is_sequence'])
def train_step(state, s, x, t, scores, masks, context_length, is_sequence=False):

    if is_sequence:
        # pad s, t with zeros
        sp = jnp.pad(s, ((0, 0), (context_length - 1, 0), (0, 0)), mode='constant', constant_values=0)
        xp = jnp.pad(x, ((0, 0), (context_length - 1, 0), (0, 0)), mode='constant', constant_values=0)
        tp = jnp.pad(t, ((0, 0), (context_length - 1, 0), (0, 0)), mode='constant', constant_values=0)

        # pad score and mask
        scorep = jnp.pad(scores, ((0, 0), (context_length - 1, 0)), mode='constant', constant_values=0)
        maskp = jnp.pad(masks, ((0, 0), (context_length - 1, 0)), mode='constant', constant_values=0)
    else:
        sp = s
        # expand x, t, scores, masks; then pad with zeros
        xp = jnp.pad(jnp.expand_dims(x, axis=1), ((0, 0), (context_length - 1, 0), (0, 0)), mode='constant', constant_values=0)
        tp = jnp.pad(jnp.expand_dims(t, axis=1), ((0, 0), (context_length - 1, 0), (0, 0)), mode='constant', constant_values=0)
        scorep = jnp.pad(jnp.reshape(scores, [-1, 1]), ((0, 0), (context_length - 1, 0)), mode='constant', constant_values=0)
        maskp = jnp.pad(jnp.reshape(masks, [-1, 1]), ((0, 0), (context_length - 1, 0)), mode='constant', constant_values=0)

    loss, grads = jitted_loss(state.params, state.model_fn, sp, xp, tp, scorep, maskp, state.get_rng())
    return state.apply_gradients(grads), loss


class Model(base.Model):

    def __init__(self, input_dims, context_length, hidden_size, layers, memory_size=16, lr=1e-4, r_seed=42):
        super().__init__("model", "transformer")

        r_key = jax.random.key(r_seed)
        self.r_key, self.dropout_r_key = jax.random.split(r_key)
        self.input_dims = input_dims
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.layers = layers
        self.r_seed = r_seed

        query_transformer = StackedTransformer(layers=layers, d_model=input_dims, output_dim=hidden_size, context_length=context_length)
        value_score_decoder = StackedTransformer(layers=layers, d_model=input_dims, output_dim=(hidden_size * memory_size * (input_dims + 1)), context_length=context_length)
        self.train_model = Value_Score_Module(records=hidden_size, slots=memory_size, output_dim=input_dims, query_backbone=query_transformer, value_score_backbone=value_score_decoder)
        self.test_model = Value_Score_Module_Test(records=hidden_size, slots=memory_size, output_dim=input_dims, query_backbone=query_transformer, value_score_backbone=value_score_decoder)

        variables = self.train_model.init(
            {'params': self.r_key, 'dropout': self.dropout_r_key}, 
            jnp.empty((1, self.context_length, input_dims), jnp.float32), 
            jnp.empty((1, self.context_length, input_dims), jnp.float32),
            jnp.empty((1, self.context_length, input_dims), jnp.float32),
        )

        self.learning_rate = lr
        self.momentum = 0.9

        optimizer = optax.adamw(self.learning_rate)
        opt_state = optimizer.init(variables['params'])

        self.state = Train_state(
            model_fn=self.train_model.apply,
            optimizer=optimizer,
            params=variables['params'],
            opt_state=opt_state,
            dropout_key=self.dropout_r_key,
            step=0
        )


    def get_class_parameters(self):
        return {
            "class_type": self.class_type,
            "class_name": self.class_name,
            "input_dims": self.input_dims,
            "context_length": self.context_length,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "memory_size": self.memory_size,
            "lr": self.learning_rate,
            "r_seed": self.r_seed
        }


    def save(self, path):
        self.state.save(path)
        self.is_updated = False


    def load(self, path):
        self.state = self.state.load(path)
        self.is_updated = False


    def fit(self, s, x, t, scores, masks=None, context=None):
        # s has shape (N, context_length, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)
        if masks is None:
            masks = jnp.ones(scores.shape)
        self.state, loss = train_step(self.state, s, x, t, scores, masks, self.context_length, is_sequence=False)
        self.is_updated = True
        return loss


    def fit_sequence(self, s, x, t, scores, masks=None, context=None):
        # s has shape (N, seq_len, dim), x has shape (N, seq_len, dim), t has shape (N, seq_len, dim), scores has shape (N, seq_len), masks has shape (N, seq_len)
        # seq_len = learning_length + context_length - 1
        if masks is None:
            masks = jnp.ones(scores.shape)
        self.state, loss = train_step(self.state, s, x, t, scores, masks, self.context_length, is_sequence=True)
        self.is_updated = True
        return loss


    def infer(self, s, t, context=None):
        # s has shape (N, context_length, dim), t has shape (N, dim)
        if s.shape[1] < self.context_length:
            # pad input
            s = jnp.pad(s, ((0, 0), (0, self.context_length - s.shape[1]), (0, 0)), mode='constant', constant_values=0)
        elif s.shape[1] > self.context_length:
            s = s[:, -self.context_length:, :]

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

    print(create_n_step_causal_mask(1, 4, 1))
    print(create_n_step_causal_mask(1, 4, 2))
    print(create_n_step_causal_mask(1, 4, 5))

    from datetime import datetime
    # get number of milliseconds since midnight of January 1, 1970
    millis = datetime.now().microsecond
    model = Model(4, 2, 8, [(4, 16), (4, 16)], 4, 0.001, r_seed=millis)

    eye = jnp.eye(4, dtype=jnp.float32)
    S = jnp.array([[eye[0, :], eye[1, :], eye[2, :], eye[3, :]]])
    X = jnp.array([[eye[3, :], eye[3, :], eye[1, :], eye[1, :]]])
    T = jnp.array([[eye[2, :], eye[2, :], eye[3, :], eye[3, :]]])
    scores = jnp.array([[0.72, 0.81, 0.9, 1.0]])

    for i in range(1000):
        loss = model.fit_sequence(S, X, T, scores)

    s = jnp.array([[eye[0, :], eye[1, :]], [eye[2, :], eye[3, :]]])
    t = jnp.array([eye[2, :], eye[3, :]])

    value, score = model.infer(s, t)
    print("Loss:", loss)
    print("Score:", score)
    print("Value:", value)
    
    # s = jnp.array([[eye[0, :]]])
    # t = jnp.array([eye[2, :]])

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
