import os
import logging
import contextlib
import random
import json
from typing import List, Any
from pydantic import BaseModel
import argparse
import sys
import math
import pickle
import jax
import jax.numpy as jnp
from jax import device_put

from utilities.utilities import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from humn import *
from implementations.jax_lm import algebraic as alg
from implementations.jax_lm import cortex, hippocampus, abstraction
import core
from core import transformer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # load llm data
    # build humn
    # train
    # save

    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")

    with open(os.path.join(experiment_path, "text_hierarchy_data.pkl"), "rb") as f:
        data = pickle.load(f)

    data_tuples = []
    item_data = data["train_set"]
    for item_datum in item_data:
        item = item_datum["item"]
        hierarchy = item_datum["hierarchy"]

        layer_paths = []
        layer_pivot_indices = []
        for i, layer in enumerate(hierarchy):
            path = alg.Embedding_Sequence(device_put(jnp.array([item_datum["start_embedding"]] + layer["embedding_chunks"], jnp.float32)))
            layer_paths.append(path)

            pivot_chunks = layer["pivot_chunks"]
            pivot_indices = alg.Pointer_Sequence([0] + [1 + p[1] for p in pivot_chunks])
            layer_pivot_indices.append(pivot_indices)

        # now add start and goal embedding again as the final top most layer
        goal_embedding = device_put(jnp.array([item_datum["goal_embedding"]], jnp.float32))
        final_pivots = alg.Embedding_Sequence(goal_embedding)
        layer_pivot_indices.append(alg.Pointer_Sequence([len(layer_paths[-1])]))
        layer_paths.append(final_pivots)

        # now zip
        layer_data = list(zip(layer_paths[:-1], layer_pivot_indices[1:], layer_paths[1:]))
        data_tuples.append(layer_data)
    

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

    embedding_dim = len(data["vocabulary"]["embeddings"][0])

    cortex_models = [
        cortex.Model(0, transformer.Model(embedding_dim, 64, 128, [64, 64])),
        cortex.Model(1, transformer.Model(embedding_dim, 64, 128, [64, 64])),
        cortex.Model(2, transformer.Model(embedding_dim, 64, 128, [64, 64]))
    ]
    hippocampus_models = [
        hippocampus.Model(64, embedding_dim),
        hippocampus.Model(64, embedding_dim),
        hippocampus.Model(64, embedding_dim)
    ]
    abstraction_models = []
    
    model = HUMN(cortex_models, hippocampus_models, abstraction_models)

    # prepare hierarchy data abstract path and train
    for path_tuples in data_tuples:
        trainers = model.observe(path_tuples)
    for trainer in trainers:
        trainer.prepare_batch(4)

    loop_train(trainers, 20000)

    # save model
    core.initialize(os.path.join(experiment_path, "core"))
    for i, (c, h) in enumerate(zip(cortex_models, hippocampus_models)):
        layer_path = os.path.join(experiment_path, "layers", f"layer_{i}")
        cortex_path = os.path.join(layer_path, "cortex")
        hippocampus_path = os.path.join(layer_path, "hippocampus")
        cortex.Model.save(c, cortex_path)
        hippocampus.Model.save(h, hippocampus_path)
        
    with open(os.path.join(experiment_path, "metadata.json"), "w") as f:
        json.dump({"num_layers": len(cortex_models), "num_abstraction_models": len(abstraction_models)}, f)