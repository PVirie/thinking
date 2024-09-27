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
from llm.src import algebric as alg
from llm.src import cortex, hippocampus, abstraction
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
    item_data = data["data"]
    for item_datum in item_data:
        item = item_datum["item"]
        hierarchy = item_datum["hierarchy"]
        start_embedding = device_put(jnp.array([item_datum["start_embedding"]], jnp.float32))
        goal_embedding = device_put(jnp.array([item_datum["goal_embedding"]], jnp.float32))

        layer_paths = []
        layer_pivots = []
        for i, layer in enumerate(hierarchy):
            # add start and goal embedding for every layer
            embedding_chunks = jnp.concatenate([start_embedding, device_put(jnp.array(layer["embedding_chunks"], jnp.float32)), goal_embedding], dim=0)
            path = alg.State_Sequence(embedding_chunks)
            layer_paths.append(path)

            pivot_chunks = layer["pivot_chunks"]
            indices = []
            for j, pivot_chunk in enumerate(pivot_chunks):
                for k in range(pivot_chunk[0], pivot_chunk[1]):
                    indices.append(j)

            pivot_indices = alg.Pointer_Sequence(indices)
            layer_pivots.append(pivot_indices)

        for i in range(len(layer_paths) - 1):
            data_tuples.append((layer_paths[i], layer_pivots[i], layer_paths[i + 1]))

        # now add start and goal embedding again as the final top most layer
        final_pivots = alg.State_Sequence(jnp.tile(jnp.expand_dims(goal_embedding, axis=0), (len(layer_paths[-1]), 1, 1)))
        data_tuples.append((layer_paths[-1], layer_pivots[-1], final_pivots))
    

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
        cortex.Model(0, transformer.Model(embedding_dim, 64, 64, [(64, 64), (64, 64)])),
        cortex.Model(1, transformer.Model(embedding_dim, 64, 64, [(64, 64), (64, 64)])),
        cortex.Model(2, transformer.Model(embedding_dim, 64, 64, [(64, 64), (64, 64)]))
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
        trainer.prepare_batch(64)

    loop_train(trainers, 100000)

    # save model
    for i, (c, h) in enumerate(zip(cortex_models, hippocampus_models)):
        layer_path = os.path.join(experiment_path, "layers", f"layer_{i}")
        cortex_path = os.path.join(layer_path, "cortex")
        hippocampus_path = os.path.join(layer_path, "hippocampus")
        cortex.Model.save(c, cortex_path)
        hippocampus.Model.save(h, hippocampus_path)
        