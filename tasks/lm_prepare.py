import contextlib
import logging
import argparse
import json
from pydantic import BaseModel
import openai
import os
os.environ['HF_HOME'] = '/app/cache/'

# use gpt-4o to generate text data
# use selected lm to token and embed text data

from utilities import *
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
