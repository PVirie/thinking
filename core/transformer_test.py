import jax
import jax.random
import jax.numpy as jnp

try:
    from . import transformer
except:
    import transformer


if __name__ == "__main__":
    print(jax.__version__)
    print(jax.devices())

    from datetime import datetime
    # get number of milliseconds since midnight of January 1, 1970
    millis = datetime.now().microsecond

    # high dims numerical stability test
    dims = 1024
    model = transformer.Model(dims, 16, 64, [64, 64], 8, r_seed=millis)

    r_key = jax.random.key(millis)
    r_key, subkey = jax.random.split(r_key)
    s = jax.random.normal(subkey, (1, 1, dims))
    r_key, subkey = jax.random.split(r_key)
    t = jax.random.normal(subkey, (128, dims))

    r_key, subkey = jax.random.split(r_key)
    S = jnp.concatenate([jnp.tile(s, (128, 1, 1)), jax.random.normal(subkey, (128, 31, dims), dtype=jnp.float32)], axis=1)
    X = jnp.roll(S, -1, axis=1)
    T = jnp.tile(jnp.expand_dims(t, axis=1), (1, 32, 1))
    r_key, subkey = jax.random.split(r_key)
    scores = jax.random.uniform(subkey, (128, 32), minval=0.0, maxval=1.0)

    for i in range(10000):
        loss = model.fit_sequence(S, X, T, scores)
        if i % 100 == 0:
            print("Loss:", loss)

    value_, score_ = model.infer(s, t[0:1, :])
    print("Score:", score_[0], "Expected:", scores[0, 0])
    print("Value:", value_[0, :], "Expected:", X[0, :])
    print("Diff:", jnp.mean(jnp.abs(value_[0, :] - X[0, :])))