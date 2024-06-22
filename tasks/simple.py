import jax.numpy as jnp
import os
import logging

from src.utilities import *
from src.jax import algebric as alg


def load(path):
    return None

def save(path, data):
    pass


def setup():
    os.makedirs(path, exist_ok=True)
    graph_shape = 16
    one_hot = generate_onehot_representation(np.arange(graph_shape), graph_shape)
    states = [alg.State(one_hot[i, :]) for i in range(16)]




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    artifact_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "artifacts")
    experiment_path = os.path.join(artifact_path, "experiments", "simple")

    exp_data = load(experiment_path)
    if exp_data is None:
        exp_data = setup()
        save(experiment_path, exp_data)