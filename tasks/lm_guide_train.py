import os
import logging
import json
from typing import List, Any
import sys
import pickle
import jax
import jax.numpy as jnp
from jax import device_put
import random

from utilities.utilities import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from humn import *
from implementations.jax_lm import algebraic as alg
from implementations.jax_lm import cortex, hippocampus, abstraction, tokenizer
import core
from core import transformer


def loop_train(trainers, num_epoch=1000):
    print_steps = max(1, num_epoch // 100)
    stamp = time.time()
    for i in range(num_epoch):
        for trainer in trainers:
            trainer.step_update()
        if i % print_steps == 0 and i > 0:
            # print at every 1 % progress
            # compute time to finish in seconds
            logging.info(f"Training progress: {(i * 100 / num_epoch):.2f}, time to finish: {((time.time() - stamp) * (num_epoch - i) / i):.2f}s")
            logging.info(f"Layer loss: {', '.join([f'{trainer.avg_loss:.4f}' for trainer in trainers])}")
    logging.info(f"Total learning time {time.time() - stamp}s")

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # load llm data
    # build humn
    # train
    # save

    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")

    with open(os.path.join(experiment_path, "text_hierarchy_data.pkl"), "rb") as f:
        data = pickle.load(f)

    item_data = data["train_set"]
    num_layers = 3
    step_size = 4
    embedding_dim = len(data["vocabulary"]["embeddings"][0])

    tokenizers = []
    fixed_embeddings = []
    full_path_data = []
    full_pivot_indices_data = []

    def check_and_add(embedding):
        found = False
        for fixed_embedding in fixed_embeddings:
            # compute cosine similarity
            score = (fixed_embedding * embedding) / (jnp.linalg.norm(fixed_embedding) * jnp.linalg.norm(embedding))
            if (score > 0.9).any():
                found = True
                break
        if not found:
            fixed_embeddings.append(embedding)

    for item_datum in item_data:
        check_and_add(jnp.array(item_datum["start_embedding"], jnp.float32))
        full_path_data.append([])
        full_pivot_indices_data.append([])

    t = tokenizer.KMean_Tokenizer(512, random.randint(0, 1000000))
    t.manually_prepend(jnp.array(data["vocabulary"]["embeddings"]))
    t.manually_prepend(fixed_embeddings)
    t.freeze()
    tokenizers.append(t)

    path_data = []
    for j, item_datum in enumerate(item_data):
        path = jnp.array(item_datum["embedding_chunks"], jnp.float32)
        path_data.append(path)
        full_path_data[j].append(t.encode(path))

    for i in range(num_layers - 1):
        next_path_data = []
        t = tokenizer.KMean_Tokenizer(512, random.randint(0, 1000000))
        for j, path in enumerate(path_data):
            pivots, pivot_indices = abstraction.process_chunk(path, step_size)
            full_pivot_indices_data[j].append(pivot_indices)
            next_path_data.append(pivots)
            t.accumulate_batch(pivots)
        t.train()
        t.manually_prepend(fixed_embeddings)
        t.freeze()
        tokenizers.append(t)

        for j, pivots in enumerate(next_path_data):
            full_path_data[j].append(t.encode(pivots))
        path_data = next_path_data

    # add final layer
    for j, item_datum in enumerate(item_data):
        full_pivot_indices_data[j].append([len(full_path_data[j][-1])])
        full_path_data[j].append(jnp.reshape(jnp.array(item_datum["goal_embedding"], jnp.float32), [1, -1]))

    cortex_models = []
    hippocampus_models = []
    for i in range(num_layers):
        input_dims = tokenizers[i].output_dims()
        target_dims = tokenizers[i + 1].output_dims() if i + 1 < num_layers else embedding_dim
        cortex_models.append(cortex.Model(i, transformer.Model([input_dims, input_dims + 1, target_dims], 4, 256, [256, 128])))
        hippocampus_models.append(hippocampus.Model(8, input_dims))

    abstraction_models = []

    model = HUMN(cortex_models, hippocampus_models, abstraction_models, max_sub_steps=16)

    for j, item_datum in enumerate(item_data):
        item = item_datum["item"]
        start_embedding = jnp.reshape(jnp.array(item_datum["start_embedding"], jnp.float32), [1, -1])

        paths = []
        indices = []
        pivots = []
        for i in range(num_layers):
            # prepend start_embedding to paths
            paths.append(alg.Embedding_Sequence(jnp.concatenate([tokenizers[i].encode(start_embedding), full_path_data[j][i]], axis = 0)))
            indices.append(alg.Pointer_Sequence(full_pivot_indices_data[j][i]))
            pivots.append(alg.Embedding_Sequence(full_path_data[j][i + 1]))

        layer_data = list(zip(paths, indices, pivots))
        trainers = model.observe(layer_data)

    for trainer in trainers:
        trainer.prepare_batch(16)

    loop_train(trainers, 50000)

    # save model
    core.initialize(os.path.join(experiment_path, "core"), clear=True)
    for i, (c, h) in enumerate(zip(cortex_models, hippocampus_models)):
        layer_path = os.path.join(experiment_path, "layers", f"layer_{i}")
        cortex_path = os.path.join(layer_path, "cortex")
        hippocampus_path = os.path.join(layer_path, "hippocampus")
        cortex.Model.save(c, cortex_path)
        hippocampus.Model.save(h, hippocampus_path)

    for i, model in enumerate(abstraction_models):
        if model is None:
            continue
        abstraction_path = os.path.join(experiment_path, "abstraction_models", f"abstraction_{i}")
        abstraction.Model.save(model, abstraction_path)

    for i, model in enumerate(tokenizers):
        tokenizer_path = os.path.join(experiment_path, "tokenizers", f"tokenizer_{i}")
        tokenizer.KMean_Tokenizer.save(model, tokenizer_path)
        
    with open(os.path.join(experiment_path, "metadata.json"), "w") as f:
        json.dump({
            "num_layers": len(cortex_models), 
            "num_abstraction_models": len(abstraction_models),
            "num_tokenizers": len(tokenizers)
        }, f)