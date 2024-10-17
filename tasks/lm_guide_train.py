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
            logging.info(f"Layer loss: {'| '.join([f'{i}, {trainer.avg_loss:.4f}' for i, trainer in enumerate(trainers)])}")
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

    full_path_data = []
    full_pivot_indices_data = []

    for item_datum in item_data:
        full_path_data.append([])
        full_pivot_indices_data.append([])

    path_data = []
    for j, item_datum in enumerate(item_data):
        path = jnp.array(item_datum["embedding_chunks"], jnp.float32)
        path_data.append(path)
        full_path_data[j].append(path)

    for i in range(num_layers - 1):
        next_path_data = []
        t = tokenizer.KMean_Tokenizer(512, random.randint(0, 1000000))
        for j, path in enumerate(path_data):
            pivots, pivot_indices = abstraction.process_chunk(path, step_size)
            full_pivot_indices_data[j].append(pivot_indices)
            full_path_data[j].append(t.encode(pivots))
        path_data = next_path_data

    # add final layer
    for j, item_datum in enumerate(item_data):
        full_pivot_indices_data[j].append([len(full_path_data[j][-1])])
        full_path_data[j].append(jnp.reshape(jnp.array(item_datum["goal_embedding"], jnp.float32), [1, -1]))

    cortex_models = [
        cortex.Model(i, transformer.Model([embedding_dim, embedding_dim + 1, embedding_dim], 4, 512, [512, 256])),
        cortex.Model(i, transformer.Model([embedding_dim, embedding_dim + 1, embedding_dim], 2, 512, [512, 256])),
        cortex.Model(i, transformer.Model([embedding_dim, embedding_dim + 1, embedding_dim], 1, 512, [512, 256]))
    ]
    hippocampus_models = [
        hippocampus.Model(64, embedding_dim),
        hippocampus.Model(64, embedding_dim),
        hippocampus.Model(64, embedding_dim)
    ]

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
            paths.append(alg.Embedding_Sequence(jnp.concatenate([start_embedding, full_path_data[j][i]], axis = 0)))
            indices.append(alg.Pointer_Sequence(full_pivot_indices_data[j][i]))
            pivots.append(alg.Embedding_Sequence(full_path_data[j][i + 1]))

        # print data
        if j == 0:
            for i, path in enumerate(paths):
                path_indices = jnp.argmax(path.data, axis=1)
                logging.info(f"Data {j} Path {i}: {path_indices}")

            for i, pivot in enumerate(pivots):
                path_indices = jnp.argmax(pivot.data, axis=1)
                logging.info(f"Data {j} Pivot {i}: {path_indices}")

        layer_data = list(zip(paths, indices, pivots))
        trainers = model.observe(layer_data)

    for trainer in trainers:
        trainer.prepare_batch(16)

    loop_train(trainers, 100000)

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

    with open(os.path.join(experiment_path, "metadata.json"), "w") as f:
        json.dump({
            "num_layers": len(cortex_models), 
            "num_abstraction_models": len(abstraction_models),
        }, f)