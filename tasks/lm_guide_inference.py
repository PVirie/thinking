import os
import logging
import json
from typing import List, Any
import sys
import pickle
import random
import jax
import jax.numpy as jnp
from jax import device_put
import numpy as np
import functools

from utilities.utilities import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from humn import *
from implementations.jax_lm import algebraic as alg
from implementations.jax_lm import cortex, hippocampus, abstraction
import core
from core import transformer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    r_key = jax.random.key(random.randint(0, 1000000))
    
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")
    core.initialize(os.path.join(experiment_path, "core"))

    with open(os.path.join(experiment_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    cortex_models = []
    hippocampus_models = []
    for i in range(metadata["num_layers"]):
        layer_path = os.path.join(experiment_path, "layers", f"layer_{i}")
        cortex_path = os.path.join(layer_path, "cortex")
        hippocampus_path = os.path.join(layer_path, "hippocampus")
        cortex_models.append(cortex.Model.load(cortex_path))
        hippocampus_models.append(hippocampus.Model.load(hippocampus_path))
    abstraction_models = []
    for i in range(metadata["num_abstraction_models"]):
        abstraction_path = os.path.join(experiment_path, "abstraction_models", f"abstraction_{i}")
        abstraction_model = None
        if os.path.exists(abstraction_path):
            abstraction_model = abstraction.Model.load(abstraction_path)
        abstraction_models.append(abstraction_model)

    model = HUMN(cortex_models, hippocampus_models, abstraction_models, max_sub_steps=128)

    def print_state(i, text_embedding):
        logging.info(f"Layer {i}: {np.linalg.norm(np.asarray(text_embedding.data)):.4f}")

    for i, c in enumerate(model.cortices):
        c.printer = functools.partial(print_state, i)

    with open(os.path.join(experiment_path, "text_hierarchy_data.pkl"), "rb") as f:
        data = pickle.load(f)

    embedding_dim = len(data["vocabulary"]["embeddings"][0])

    data_tuples = []
    item_data = data["train_set"]
    # item_data = data["test_set"]
    for item_datum in item_data:
        item = item_datum["item"]
        logging.log(logging.INFO, f"Processing {item}...")
        start = alg.Text_Embedding(item_datum["start_embedding"])
        goal = alg.Text_Embedding(item_datum["goal_embedding"])

        chunks = []
        try:
            for state in model.think(start, goal):
                numpy_state = np.asarray(state.data)
                chunks.append(numpy_state)
        except MaxSubStepReached:
            logging.warning(f"Truncated termination for item {item}.")
        finally:
            logging.info(", ".join([f"{np.linalg.norm(chunk):.4f}" for chunk in item_datum["embedding_chunks"]]))
            logging.info(", ".join([f"{np.linalg.norm(chunk):.4f}" for chunk in chunks]))

        data_tuples.append({
            "embedding_chunks": [chunk.tolist() for chunk in chunks],
        })

    with open(os.path.join(experiment_path, "guide_results.pkl"), "wb") as f:
        pickle.dump(data_tuples, f)

    logging.info("Done.")
        