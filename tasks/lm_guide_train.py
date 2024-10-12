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
from implementations.jax_lm import cortex, hippocampus, abstraction
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

    cortex_models = [
        cortex.Model(i, transformer.Model(embedding_dim, 4, 128, [128, 64]))
        for i in range(num_layers)
    ]
    hippocampus_models = [
        hippocampus.Model(64, embedding_dim, 128, random.randint(0, 1000000))
        for i in range(num_layers)
    ]
    abstraction_models = [
        abstraction.Model(step_size)
        for i in range(num_layers - 1)
    ]
    
    fixed_embeddings = [
        [0.0] * embedding_dim
    ]

    full_path_data = []
    full_pivot_indices_data = []
    path_data = []
    for item_datum in item_data:
        fixed_embeddings.append(item_datum["start_embedding"])
        embedding_chunks = jnp.array(item_datum["embedding_chunks"], jnp.float32)
        path = alg.Embedding_Sequence(embedding_chunks)
        path_data.append(path)
        full_path_data.append([path])
        full_pivot_indices_data.append([])
        
    fixed_embeddings = jnp.array(fixed_embeddings)
    hippocampus_models[0].trainer.manually_append(jnp.array(data["vocabulary"]["embeddings"]))
    hippocampus_models[0].trainer.manually_append(fixed_embeddings)

    for i in range(num_layers - 1):
        pivots_temp = []
        for j, path in enumerate(path_data):
            pivot_indices, pivots = abstraction_models[i].abstract_path(path)
            full_pivot_indices_data[j].append(pivot_indices)
            pivots_temp.append(pivots)
            trainer = hippocampus_models[i+1].incrementally_learn(pivots)
        trainer.train()
        trainer.manually_append(fixed_embeddings)

        next_path_data = []
        for j, pivots in enumerate(pivots_temp):
            path = alg.Embedding_Sequence(hippocampus_models[i+1].refine(pivots.data))
            full_path_data[j].append(path)
            next_path_data.append(path)
        path_data = next_path_data

    # add final layer
    for j in range(len(item_data)):
        full_path_data[j].append(alg.Embedding_Sequence())
        full_pivot_indices_data[j].append(alg.Pointer_Sequence())

    model = HUMN(cortex_models, hippocampus_models, abstraction_models, reset_hippocampus_on_target_changed=True, max_sub_steps=16)

    stop_embedding = alg.Text_Embedding(jnp.zeros([embedding_dim], jnp.float32))
    for j, item_datum in enumerate(item_data):
        item = item_datum["item"]
        start_embedding = alg.Text_Embedding(jnp.array(item_datum["start_embedding"], jnp.float32))
        goal_embedding = alg.Text_Embedding(jnp.array(item_datum["goal_embedding"], jnp.float32))

        # prepend start_embedding and append stop_embedding to path
        # append len(path) to pivot_indices (offset start_embedding)
        # append goal_embedding to pivots

        paths = [x.pre_append(start_embedding, stop_embedding) for x in full_path_data[j][:-1]]
        indices = [x.append(len(full_path_data[j][i])) for i, x in enumerate(full_pivot_indices_data[j])]
        pivots = [x.append(goal_embedding) for x in full_path_data[j][1:]]

        layer_data = list(zip(paths, indices, pivots))
        trainers = model.observe(layer_data)

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

    for i, model in enumerate(abstraction_models):
        if model is None:
            continue
        abstraction_path = os.path.join(experiment_path, "abstraction_models", f"abstraction_{i}")
        abstraction.Model.save(model, abstraction_path)
        
    with open(os.path.join(experiment_path, "metadata.json"), "w") as f:
        json.dump({"num_layers": len(cortex_models), "num_abstraction_models": len(abstraction_models)}, f)