import jax.numpy as jnp
import jax
from jax import random
from jax import jit
import time

print(jax.devices())

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

key = random.key(1701)
x = jax.device_put(random.normal(key, (1_000_000,)))


start_time = time.time()
for i in range(1000):
    selu(x).block_until_ready()
print(f"SELUs took {time.time() - start_time:.2f} seconds")


selu_jit = jit(selu)
selu_jit(x)  # compiles on first call

start_time = time.time()
for i in range(1000):
    selu_jit(x).block_until_ready()
print(f"JITted SELUs took {time.time() - start_time:.2f} seconds")