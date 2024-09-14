import contextlib
import logging
import argparse
import json
from pydantic import BaseModel

from utilities.utilities import *
from utilities.lm.huggingface_lm import Model as Small_Model
from utilities.lm.openai_lm import Model as Large_Model


# use large model to generate text data
# use small model tokenize and embed text data

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import core


class Context(BaseModel):
    random_seed: int = 0

    @staticmethod
    def load(path):
        return None
                                 

    @staticmethod
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        return True


    @staticmethod
    def setup(setup_path):
        pass


@contextlib.contextmanager
def experiment_session(path, force_clear=None):
    core.initialize(os.path.join(path, "core"))
    if force_clear is not None and force_clear:
        logging.info(f"Clearing the experiment directory: {path}")
        empty_directory(path)
    context = Context.load(path)
    if context is None:
        context = Context.setup(path)
        Context.save(context, path)
    yield context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm")
