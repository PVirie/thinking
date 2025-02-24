import os
from typing import Sequence, Union, List, Any
from collections.abc import Callable
import jax
import jax.random
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax import core, struct              # Flax dataclasses
import optax
import pickle
from functools import partial

try:
    from . import base
except:
    import base


@partial(jax.jit, static_argnames=['batch_size', 'seq_len', 'n'])
def create_n_step_causal_mask(batch_size, seq_len, n):
    # unlike usual causal mask, this mask will have n steps backward only
    # for example n = 2
    # [1, 0, 0, 0]
    # [1, 1, 0, 0]
    # [0, 1, 1, 0]
    # [0, 0, 1, 1]

    n = jnp.minimum(n, seq_len)

    def body_fun(i, result):
        mask = jnp.eye(seq_len, dtype=jnp.int32)
        mask = jnp.roll(mask, -i, axis=1)
        mask = jnp.tril(mask)
        return result + mask
    
    mask = jnp.zeros((seq_len, seq_len), dtype=jnp.int32)
    mask = lax.fori_loop(0, n, body_fun, mask)

    # Convert the mask to shape (batch_size, 1, seq_len, seq_len) for broadcasting
    mask = jnp.expand_dims(mask, axis=0)  # Shape: (1, seq_len, seq_len)
    mask = jnp.expand_dims(mask, axis=0)  # Shape: (batch_size, 1, seq_len, seq_len)
    mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))

    return mask


@partial(jax.jit, static_argnames=['batch_size', 'seq_len'])
def create_self_attention_mask(batch_size, seq_len):
    mask = jnp.eye(seq_len)
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
    layers: Sequence[int]
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
        for d_ff in self.layers:
            decoder_input = TransformerDecoderBlock(
                d_model=d_model,
                num_heads=d_model,
                d_ff=d_ff,
                dropout_rate=self.dropout_rate
            )(encoder_input, decoder_input, mask, train)

        # Final layer norm and projection to output dimension
        decoder_input = nn.LayerNorm(epsilon=1e-6)(decoder_input)
        decoder_output = nn.Dense(self.output_dim)(decoder_input)

        return decoder_output


class Value_Score_Module(nn.Module):
    slots: int
    output_dim: int
    value_score_backbone: StackedTransformer
    value_access: bool = True
    # records: int
    # query_backbone: StackedTransformer

    @nn.compact
    def __call__(self, s, x, t, scores):
        seq_len = s.shape[1]

        Wvs = self.value_score_backbone(t, s, True)
        Vs = Wvs[:, :, :self.slots * self.output_dim]
        Ss = Wvs[:, :, self.slots * self.output_dim:]

        Vs = jnp.reshape(Vs, (-1, self.slots, self.output_dim))
        Ss = jnp.reshape(Ss, (-1, self.slots))

        # Access
        if self.value_access:
            x = jnp.reshape(x, (-1, self.output_dim))
            Vl = jnp.expand_dims(x, axis=2)
            denom = jnp.linalg.norm(Vs, axis=2, keepdims=True) * jnp.linalg.norm(Vl, axis=1, keepdims=True)
            # prevent divide by zero
            denom = jnp.maximum(1e-6, denom)
            dot_scores = jnp.linalg.matmul(Vs, Vl) / denom
            # force dot_scores shape [batch, memory_size]
            dot_scores = jnp.reshape(dot_scores, (-1, self.slots))
            max_indices = jnp.argmax(dot_scores, axis=1, keepdims=True)    
        else:
            # score has range [0, 1], quantize to int slots, must be handled from out side
            scores = jnp.reshape(scores, (-1, 1))
            max_indices = jnp.round(scores * (self.slots - 1)).astype(jnp.int32)
            max_indices = jnp.clip(max_indices, min=0, max=self.slots - 1)

        s = jnp.take_along_axis(Ss, max_indices, axis=1)
        v = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
        
        v = jnp.reshape(v, (-1, seq_len, self.output_dim))
        s = jnp.reshape(s, (-1, seq_len))

        return v, s, Ss


class Value_Score_Module_Test(Value_Score_Module):
    @nn.compact
    def __call__(self, s, t):
        
        Wvs = self.value_score_backbone(t, s, False)
        Vs = Wvs[:, -1, :self.slots * self.output_dim]
        Ss = Wvs[:, -1, self.slots * self.output_dim:]
        
        Vs = jnp.reshape(Vs, (-1, self.slots, self.output_dim))
        Ss = jnp.reshape(Ss, (-1, self.slots))

        # Access

        max_indices = jnp.argmax(Ss, axis=1, keepdims=True)
        s = jnp.take_along_axis(Ss, max_indices, axis=1)
        v = jnp.take_along_axis(Vs, jnp.expand_dims(max_indices, axis=-1), axis=1)
        
        return v, s

    
class Train_state(struct.PyTreeNode):
    model_fn: Callable = struct.field(pytree_node=False)
    inference_fn: Callable = struct.field(pytree_node=False)
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
    v_, scores_, Ss = model_fn({'params': params}, s, x, t, scores, rngs={'dropout': dropout_train_key})
    
    # prevent div by zero
    sum_mask = jnp.sum(masks)
    sum_mask = jnp.maximum(sum_mask, 1)
    error_S = jnp.sum(masks * (scores - scores_)**2) / sum_mask
    error_V = jnp.sum(masks * jnp.mean((x - v_)**2, axis=-1)) / sum_mask
    
    # suppress other slot score to mean of overall
    # loss = objective_loss + normalizer = value_loss + score_loss + log sum exp (all score)
    # d loss/ d param = d value_loss/ d param + d score_loss/ d param + sum (prop . d score / d param)
    error_C = jnp.mean(jax.nn.logsumexp(Ss, axis=1))

    return error_V + error_S + error_C * 0.01


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


@partial(jax.jit, static_argnames=['context_length', 'input_dims', 'target_dims'])
def execute_fn(state, s, t, context_length, input_dims, target_dims):

    s = jnp.reshape(s, (-1, context_length, input_dims))
    t = jnp.reshape(t, (-1, 1, target_dims))
    t = jnp.repeat(t, context_length, axis=1)

    best_value, best_score = state.inference_fn({'params': state.params}, s, t, mutable=False, rngs={'dropout': state.dropout_key})

    return best_value, best_score


class Model(base.Model):

    def __init__(self, dims, context_length, hidden_size, layers, memory_size=16, value_access=True, lr=1e-4, r_seed=42):
        super().__init__("model", "transformer")

        # if dims is an integer, then dims is the number of dimensions
        self.dims = dims
        if isinstance(dims, int):
            self.input_dims = dims
            self.next_state_dims = dims
            self.target_dims = dims
        else:
            self.input_dims = dims[0]
            self.next_state_dims = dims[1]
            self.target_dims = dims[2]

        r_key = jax.random.key(r_seed)
        self.r_key, self.dropout_r_key = jax.random.split(r_key)
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.value_access = value_access
        self.layers = layers
        self.r_seed = r_seed

        # query_transformer = StackedTransformer(layers=layers, d_model=self.input_dims, output_dim=hidden_size, context_length=context_length)
        value_score_decoder = StackedTransformer(layers=layers, d_model=hidden_size, output_dim=memory_size * (self.next_state_dims + 1), context_length=context_length)
        self.train_model = Value_Score_Module(slots=memory_size, output_dim=self.next_state_dims, value_score_backbone=value_score_decoder, value_access=self.value_access)
        self.test_model = Value_Score_Module_Test(slots=memory_size, output_dim=self.next_state_dims, value_score_backbone=value_score_decoder, value_access=self.value_access)

        variables = self.train_model.init(
            {'params': self.r_key, 'dropout': self.dropout_r_key}, 
            jnp.empty((1, self.context_length, self.input_dims), jnp.float32), 
            jnp.empty((1, self.context_length, self.next_state_dims), jnp.float32),
            jnp.empty((1, self.context_length, self.target_dims), jnp.float32),
            jnp.empty((1, self.context_length), jnp.float32)
        )

        self.learning_rate = lr
        self.momentum = 0.9

        optimizer = optax.adamw(self.learning_rate)
        opt_state = optimizer.init(variables['params'])

        self.state = Train_state(
            model_fn=self.train_model.apply,
            inference_fn=self.test_model.apply,
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
            "dims": self.dims,
            "context_length": self.context_length,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "memory_size": self.memory_size,
            "value_access": self.value_access,
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
        # s has shape (N, context_length, input_dims), x has shape (N, next_state_dims), t has shape (N, target_dims), scores has shape (N), masks has shape (N)
        if masks is None:
            masks = jnp.ones(scores.shape)
        self.state, loss = train_step(self.state, s, x, t, scores, masks, self.context_length, is_sequence=False)
        self.is_updated = True
        return loss


    def fit_sequence(self, s, x, t, scores, masks=None, context=None):
        # s has shape (N, seq_len, input_dims), x has shape (N, seq_len, next_state_dims), t has shape (N, seq_len, target_dims), scores has shape (N, seq_len), masks has shape (N, seq_len)
        if masks is None:
            masks = jnp.ones(scores.shape)
        self.state, loss = train_step(self.state, s, x, t, scores, masks, self.context_length, is_sequence=True)
        self.is_updated = True
        return loss


    def infer(self, s, t, context=None):
        # s has shape (N, context_length, input_dims), t has shape (N, target_dims)
        seq_len = s.shape[1]
        if seq_len < self.context_length:
            # pad input
            s = jnp.pad(s, ((0, 0), (self.context_length - s.shape[1], 0), (0, 0)), mode='constant', constant_values=0)
        elif seq_len > self.context_length:
            s = s[:, -self.context_length:, :]

        best_value, best_score = execute_fn(self.state, s, t, self.context_length, self.input_dims, self.target_dims)

        best_value = best_value[:, -1, :]
        best_score = best_score[:, -1]
        return best_value, best_score



if __name__ == "__main__":

    print(create_n_step_causal_mask(1, 4, 1))
    print(create_n_step_causal_mask(1, 4, 2))
    print(create_n_step_causal_mask(1, 4, 5))

    from datetime import datetime
    # get number of milliseconds since midnight of January 1, 1970
    millis = datetime.now().microsecond
    model = Model(4, 2, 8, [16, 16], 4, lr=0.001, r_seed=millis)

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
