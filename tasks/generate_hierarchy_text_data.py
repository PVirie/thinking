import os
os.environ['HF_HOME'] = '/app/cache/'

import jax
from jax import random
import jax.numpy as jnp

print(jax.devices())

# model_id = "meta-llama/Meta-Llama-3-8B"

# pipeline = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
# pipeline("Hey how are you doing today?")

# tokenizer = LlamaTokenizer.from_pretrained(model_id)
# model = FlaxLlamaForCausalLM.from_pretrained(model_id)

# inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

# for i in range(10):
#     outputs = model(**inputs)

#     # retrieve logts for next token
#     next_token_logits = outputs.logits[:, -1]

#     # decode outputs
#     next_token = next_token_logits.argmax(-1)

#     # update input_ids
#     inputs["input_ids"] = jnp.concatenate([inputs["input_ids"], jnp.expand_dims(next_token, axis=0)], axis=-1)
#     inputs["attention_mask"] = jnp.ones_like(inputs["input_ids"])

#     results = tokenizer.decode(inputs["input_ids"][0])
#     print(results)