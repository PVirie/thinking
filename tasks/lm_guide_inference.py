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
from implementations.jax_lm import algebric as alg
from implementations.jax_lm import cortex, hippocampus, abstraction
import core
from core import transformer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")

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

    model = HUMN(cortex_models, hippocampus_models, abstraction_models)
