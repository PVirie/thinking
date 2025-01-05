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
    for i in range(10000):
        selu(x).block_until_ready()
    print(f"SELUs took {time.time() - start_time:.4f} seconds")

    # test jax jit

    selu_jit = jit(selu)
    selu_jit(x)  # compiles on first call

    start_time = time.time()
    for i in range(10000):
        selu_jit(x).block_until_ready()
    print(f"JITted SELUs took {time.time() - start_time:.4f} seconds")

    # test jax jit with more complex function

    def loop_selu(x):
        sum = jnp.zeros((1_000_000,))
        for i in range(1000):
            sum = sum + selu(x)
        return jnp.sum(sum)
    
    loop_selu_jit = jit(loop_selu)
    loop_selu_jit(x)  # compiles on first call

    start_time = time.time()
    loop_selu_jit(x).block_until_ready()
    print(f"JITted loop SELUs took {time.time() - start_time:.4f} seconds")

    # jax is comparable to torch.compile, but it jits functions way faster.


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

    def selu(x, alpha:float=1.67, lmbda:float=1.05):
        return lmbda * torch.where(x > 0, x, alpha * torch.exp(x) - alpha)

    x = torch.randn(1_000_000, device=device)

    start_time = time.time()
    for i in range(10000):
        selu(x)
    print(f"SELUs took {time.time() - start_time:.4f} seconds")

    # test torch legacy jit.script

    selu_jit = torch.jit.script(selu)

    start_time = time.time()
    for i in range(10000):
        selu_jit(x)
    print(f"JITted SELUs took {time.time() - start_time:.4f} seconds")
    
    # test torch jit with more complex function

    def loop_selu(x):
        sum = torch.zeros(1_000_000, device=x.device)
        for i in range(1000):
            sum = sum + selu(x)
        return torch.sum(sum)
    
    loop_selu_jit = torch.jit.script(loop_selu)

    start_time = time.time()
    loop_selu_jit(x)
    print(f"JITted loop SELUs took {time.time() - start_time:.4f} seconds")


    # test torch compile

    loop_selu_compiled = torch.compile(loop_selu)
    loop_selu_compiled(x) # compiles on first call

    start_time = time.time()
    loop_selu_compiled(x)
    print(f"Compiled loop SELUs took {time.time() - start_time:.4f} seconds")

print("Done.")