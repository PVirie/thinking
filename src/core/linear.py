import jax.random
import jax
import jax.numpy as jnp
import os

if __name__ == "__main__":
    import base
else:
    from . import base


def compute_value_score(Q, K, Wv, Ws):
    logit = jnp.matmul(Q, jnp.transpose(K))
    return jnp.matmul(logit, Wv), jnp.matmul(logit, Ws)


def compute_error(Q, V, S, M, K, Wv, Ws):
    V_, S_ = compute_value_score(Q, K, Wv, Ws)
    update_indices = (S > S_) * M
    error_V = M * (V - V_)**2
    error_S = M * (S - S_)**2
    return jnp.mean(error_V) + jnp.mean(error_S)

# extremely faster with jit
value_grad_function = jax.jit(jax.value_and_grad(compute_error, argnums=(4, 5, 6)))


class Model(base.Model):

    def __init__(self, hidden_size, dims):
        self.class_name = "linear"
        self.hidden_size = hidden_size
        self.input_dims = dims

        self.key = jax.random.normal(jax.random.PRNGKey(0), (hidden_size, dims * 2))*0.01
        self.score = jnp.zeros([hidden_size, 1], jnp.float32)
        self.value = jnp.zeros([hidden_size, dims], jnp.float32)

        self.lr = 0.1
        self.epoch_size = 100


    def get_class_parameters(self):
        return {"class_name": self.class_name, "input_dims": self.input_dims, "hidden_size": self.hidden_size}


    def save(self, path):
        jnp.save(os.path.join(path, "key.npy"), self.key)
        jnp.save(os.path.join(path, "score.npy"), self.score)
        jnp.save(os.path.join(path, "value.npy"), self.value)


    @staticmethod
    def load(path, class_parameters):
        model = Model(class_parameters["input_dims"])
        model.key = jnp.load(os.path.join(path, "key.npy"))
        model.score = jnp.load(os.path.join(path, "score.npy"))
        model.value = jnp.load(os.path.join(path, "value.npy"))
        return model


    def fit(self, s, x, t, scores, masks=1.0):
        # s has shape (N, dim), x has shape (N, dim), t has shape (N, dim), scores has shape (N), masks has shape (N)

        scores = jnp.reshape(scores, (-1, 1))
        masks = jnp.reshape(masks, (-1, 1))
        x = jnp.reshape(x, (-1, self.input_dims))

        batch = jnp.concatenate([s, t], axis=-1)
        query = jnp.reshape(batch, (-1, self.input_dims * 2))

        for i in range(self.epoch_size):
            loss, (g_K, g_Wv, g_Ws) = value_grad_function(query, x, scores, masks, self.key, self.value, self.score)

            # now update
            self.key = self.key - self.lr*g_K
            self.score = self.score - self.lr*g_Ws
            self.value = self.value - self.lr*g_Wv

        # # score_updates = (1 - update_indices) * best_score + update_indices * scores
        # self.score = 0.95*self.score + 0.05*score_updates

        # # value_updates = (1 - update_indices) * best_value + update_indices * x
        # self.value = 0.95*self.value + 0.05*value_updates

        return loss


    def infer(self, s, t):
        # s has shape (N, dim), t has shape (N, dim)

        # for simple model only use the last state
        batch = jnp.concatenate([s, t], axis=-1)
        query = jnp.reshape(batch, (-1, self.input_dims * 2))

        best_value, best_score = compute_value_score(query, self.key, self.value, self.score)

        return best_value, best_score




if __name__ == "__main__":
    model = Model(8, 4)

    eye = jnp.eye(4, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    x = jnp.array([eye[1, :], eye[2, :]])
    t = jnp.array([eye[3, :], eye[3, :]])

    loss = model.fit(s, x, t, jnp.array([1, 0]))
    score, value = model.infer(s, t)
    print(loss, score, value)
    
    loss = model.fit(s, x, t, jnp.array([0, 1]))
    score, value = model.infer(s, t)
    print(loss, score, value)