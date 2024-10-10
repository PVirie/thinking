import os
import logging
import json
from typing import List, Any
import sys
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


def average_embeddings(embedding_chunks):
    return jnp.mean(embedding_chunks, axis=0)


def process_chunk(embedding_chunks, processor, step_size=4):
    # split chunks into sub_chunk of step_size
    abstract_chunks = jnp.stack([processor(embedding_chunks[i:i+step_size, :]) for i in range(0, len(embedding_chunks), step_size)], axis=0)
    abstract_pivot_chunks = [[i, min(i + step_size, len(embedding_chunks))] for i in range(0, len(embedding_chunks), step_size)]
    return abstract_chunks, abstract_pivot_chunks
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # load llm data
    # build humn
    # train
    # save

    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")

    with open(os.path.join(experiment_path, "text_hierarchy_data.pkl"), "rb") as f:
        data = pickle.load(f)

    num_layers = 3
    step_size = 4

    embedding_dim = len(data["vocabulary"]["embeddings"][0])
    stop_embedding = jnp.zeros([1, embedding_dim], jnp.float32)

    data_tuples = []
    item_data = data["train_set"]
    for item_datum in item_data:

        item = item_datum["item"]
        start_embedding = jnp.array([item_datum["start_embedding"]], jnp.float32)
        goal_embedding = jnp.array([item_datum["goal_embedding"]], jnp.float32)
        embedding_chunks = jnp.array(item_datum["embedding_chunks"], jnp.float32)
        pivot_chunks = item_datum["pivot_chunks"]

        layer_paths = []
        layer_pivot_indices = []
        for i in range(num_layers):

            # prepend with the start embedding
            path = alg.Embedding_Sequence(jnp.concatenate([start_embedding, embedding_chunks, stop_embedding], axis=0))
            layer_paths.append(path)

            pivot_indices = alg.Pointer_Sequence([0] + [p[1] for p in pivot_chunks])
            layer_pivot_indices.append(pivot_indices)

            embedding_chunks, pivot_chunks = process_chunk(embedding_chunks, average_embeddings, step_size=step_size)

        # add the final pivot
        layer_pivot_indices.append(alg.Pointer_Sequence([len(layer_paths[num_layers-1]) - 1]))
        final_pivots = alg.Embedding_Sequence(jnp.concatenate([goal_embedding, stop_embedding], axis=0))
        layer_paths.append(final_pivots)

        # now zip (path, pivot_index, next_path for pivot (removed the stop embedding))
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


    cortex_models = [
        cortex.Model(i, transformer.Model(embedding_dim, 4, 128, [128, 64]))
        for i in range(num_layers)
    ]
    hippocampus_models = [
        hippocampus.Model(64, embedding_dim)
        for i in range(num_layers)
    ]
    abstraction_models = []
    
    model = HUMN(cortex_models, hippocampus_models, abstraction_models, reset_hippocampus_on_target_changed=True, max_sub_steps=16)

    # prepare hierarchy data abstract path and train
    for path_tuples in data_tuples:
        trainers = model.observe(path_tuples)
    for trainer in trainers:
        trainer.prepare_batch(4)

    loop_train(trainers, 100000)

    # save model
    core.initialize(os.path.join(experiment_path, "core"), clear=True)
    for i, (c, h) in enumerate(zip(cortex_models, hippocampus_models)):
        layer_path = os.path.join(experiment_path, "layers", f"layer_{i}")
        cortex_path = os.path.join(layer_path, "cortex")
        hippocampus_path = os.path.join(layer_path, "hippocampus")
        cortex.Model.save(c, cortex_path)
        hippocampus.Model.save(h, hippocampus_path)
        
    with open(os.path.join(experiment_path, "metadata.json"), "w") as f:
        json.dump({"num_layers": len(cortex_models), "num_abstraction_models": len(abstraction_models)}, f)