import time

test_jax = False
try:
    import jax.numpy as jnp
    import jax
    from jax import random
    from jax import jit
    test_jax = True
except ImportError:
    pass


if test_jax:
    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())

    def selu(x, alpha=1.67, lmbda=1.05):
        return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

    key = random.key(1701)
    x = random.normal(key, (1_000_000,))

    start_time = time.time()
    for i in range(1000):
        selu(x).block_until_ready()
    print(f"SELUs took {time.time() - start_time:.2f} seconds")

    # test jax jit

    selu_jit = jit(selu)
    selu_jit(x)  # compiles on first call

    start_time = time.time()
    for i in range(1000):
        selu_jit(x).block_until_ready()
    print(f"JITted SELUs took {time.time() - start_time:.2f} seconds")


test_torch = False
try:
    import torch
    test_torch = True
except ImportError:
    pass

if test_torch:
    print("Torch version:", torch.__version__)
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    print("Device:", device)

    def selu(x, alpha=1.67, lmbda=1.05):
        return lmbda * torch.where(x > 0, x, alpha * torch.exp(x) - alpha)

    x = torch.randn(1_000_000, device=device)

    start_time = time.time()
    for i in range(1000):
        selu(x)
    print(f"SELUs took {time.time() - start_time:.2f} seconds")

    # test torch legacy jit.script

    @torch.jit.script
    def selu_jit(x, alpha:float = 1.67, lmbda:float = 1.05):
        return lmbda * torch.where(x > 0, x, alpha * torch.exp(x) - alpha)

    start_time = time.time()
    for i in range(1000):
        selu_jit(x)
    print(f"JITted SELUs took {time.time() - start_time:.2f} seconds")

    # test torch compile

    selu_compiled = torch.compile(selu)

    start_time = time.time()
    for i in range(1000):
        selu_compiled(x)
    print(f"Compiled SELUs took {time.time() - start_time:.2f} seconds")
    

print("Done.")