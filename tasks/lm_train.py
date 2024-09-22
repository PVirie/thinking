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
import jax.numpy

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


    embedding_dim = 64

    cortex_models = [
        cortex.Model(0, transformer.Model(64, embedding_dim, 16, [(64, 64), (64, 64)])),
        cortex.Model(1, transformer.Model(64, embedding_dim, 16, [(64, 64), (64, 64)])),
        cortex.Model(2, transformer.Model(64, embedding_dim, 16, [(64, 64), (64, 64)]))
    ]
    hippocampus_models = [
        hippocampus.Model(16, embedding_dim),
        hippocampus.Model(16, embedding_dim),
        hippocampus.Model(16, embedding_dim)
    ]
    abstraction_models = []
    
    model = HUMN(cortex_models, hippocampus_models, abstraction_models)

    # prepare hierarchy data abstract path and train
    for path_tuples in data_abstract_path:
        trainers = model.observe(path_tuples)
    for trainer in trainers:
        trainer.prepare_batch(64)

    loop_train(trainers, 100000)


    # save model