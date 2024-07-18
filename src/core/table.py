import jax.numpy as jnp
import os

class Model:

    def __init__(self, dims):
        self.input_dims = dims
        # make [[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0] ...]]
        eye = jnp.eye(dims, dtype=jnp.float32)
        eye_1 = jnp.expand_dims(eye, axis=0)
        eye_1 = jnp.tile(eye_1, (dims, 1, 1))
        eye_2 = jnp.expand_dims(eye, axis=1)
        eye_2 = jnp.tile(eye_2, (1, dims, 1))
        self.key = jnp.concatenate([eye_2, eye_1], axis=-1)
        self.key = jnp.reshape(self.key, (-1, dims * 2))

        # self.key = jnp.reshape(features, (-1, self.input_dims * 2))
        self.value = jnp.zeros([dims * dims], jnp.float32)


    @staticmethod
    def make_features(s, t):
        s_ = jnp.expand_dims(s, axis=1)
        s_ = jnp.tile(s_, (1, t.shape[0], 1))

        t_ = jnp.expand_dims(t, axis=0)
        t_ = jnp.tile(t_, (s.shape[0], 1, 1))

        features = jnp.concatenate([s_, t_], axis=-1)
        return features


    def fit(self, s, t, labels, masks):
        # s has shape (N, dim), t has shape (M, dim), labels has shape (N, M), masks has shape (N, M)

        features = Model.make_features(s, t)
        batch = jnp.reshape(features, (-1, self.input_dims * 2))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        # logits has shape [NxM, len(key)]
        # find max key for each batch
        argmax_logits = jnp.argmax(logits, axis=1)

        masks = jnp.reshape(masks, (-1))
        labels = jnp.reshape(labels, (-1))

        updates = jnp.where(labels * masks > self.value[argmax_logits], labels, self.value[argmax_logits])
        # update value at argmax_logits with updates
        self.value = self.value.at[argmax_logits].set(updates)

        return jnp.mean(updates)


    def infer(self, s, t):
        # s has shape (N, dim), t has shape (M, dim), labels has shape (N, M), masks has shape (N, M)

        features = Model.make_features(s, t)
        batch = jnp.reshape(features, (-1, self.input_dims * 2))

        # access key
        logits = jnp.matmul(batch, jnp.transpose(self.key))
        # logits has shape [N*M, len(key)]
        # find max key for each batch
        argmax_logits = jnp.argmax(logits, axis=1)
        # access value, return with shape (N, M)
        return jnp.reshape(self.value[argmax_logits], features.shape[:-1])


    def save(self, path):
        jnp.save(os.path.join(path, "ideal.npy"), self.value)


    def load(self, path):
        self.value = jnp.load(os.path.join(path, "ideal.npy"))




if __name__ == "__main__":
    model = Model(3)
    print(model.key)

    eye = jnp.eye(3, dtype=jnp.float32)
    s = jnp.array([eye[0, :], eye[1, :]])
    t = eye

    model.fit(s, t, 0.5, jnp.array([1, 1, 0, 0, 0, 0]))
    value = model.infer(s, t)
    print(value)
    
    model.fit(s, t, 1.0, jnp.array([1, 0, 1, 1, 0, 0]))
    value = model.infer(s, t)
    print(value)