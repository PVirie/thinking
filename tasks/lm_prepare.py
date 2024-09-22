"""
This script is used to prepare the data for the language model experiment.
I use large model to generate text data.
I use small model to tokenize and embed text data.

"""
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

from utilities.utilities import *
from utilities.lm.huggingface_lm import Model as Small_Model
from utilities.lm.openai_lm import Model as Large_Model

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from humn import *


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")

    # 1. clear existing data
    # 2. prepare parameters
    # 3. generate text
    # 4. tokenize text
    # 5. split into hierarchy
    # 6. embedding text
    # 7. save

    logging.info(f"Clearing the experiment directory: {path}")
    empty_directory(path)

    item_list = [
        "Iron plate",
        "Copper plate",
        "Steel plate",
        "Stone brick",
        "Iron plate",
        "Copper plate",
        "Steel plate",
        "Stone brick",
        "Iron plate",
        "Copper plate",
        "Steel plate",
        "Stone brick",
        "Iron gear wheel",
        "Automation science pack",
        "Copper cable",
        "Electronic circuit",
        "Inserter",
        "Transport belt",
        "Logistic science pack",
        "Granade",
        "Firearm magazine",
        "Piercing rounds magazine",
        "Wall",
        "Military science pack",
        "Plastic bar",
        "Advanced circuit",
        "Pipe",
        "Engine unit",
        "Sulfur",
        "Chemical science pack",
        "Sulfuric acid",
        "Electric furnace",
        "Productivity module",
        "Iron stick",
        "Rail",
        "Production science pack",
        "Battery",
        "Electric engine unit",
        "Flying robot frame",
        "Low density structure",
        "Processing unit",
        "Utility science pack",
        "Solid fuel",
        "Solid fuel",
        "Solid fuel",
        "Rocket fuel",
        "Speed module",
        "Rocket control unit",
        "Rocket part",
        "Construction robot",
        "Logistic robot",
        "Fast transport belt",
        "Express transport belt",
        "Satellite",
        "Radar",
        "Accumulator",
        "Solar panel",
        "Assembling machine 1",
        "Assembling machine 2",
        "Assembling machine 3",
        "Splitter",
        "Fast splitter",
        "Express splitter",
        "Underground belt",
        "Fast underground belt",
        "Express underground belt",
        "Long-handed inserter",
        "Fast inserter",
        "Filter inserter",
        "Stack inserter",
        "Stack filter inserter",
    ]

    sentence_format = "In order to build {} in factorio, here are the steps:"
    large_model = Large_Model()
    small_model = Small_Model()
    step_size = 10
    num_layers = 3


    def average_embeddings(embedding_chunks):
        return Small_Model.compute_mean_embedding(embedding_chunks)


    def process_chunk(embedding_chunks, processor):
        # split chunks into sub_chunk of step_size
        abstract_chunks = [processor(embedding_chunks[i:i+step_size]) for i in range(0, len(embedding_chunks), step_size)]
        abstract_pivot_chunks = [[i, min(i + step_size, len(embedding_chunks))] for i in range(0, len(embedding_chunks), step_size)]
        return abstract_chunks, abstract_pivot_chunks
        
    data = []
    for item_i, item in enumerate(item_list):
        if item_i % 10 == 0:
            logging.info(f"Processing item {item_i} out of {len(item_list)}")

        query = sentence_format.format(item)
        text_response = large_model.get_chat_response(query, token_length=2000)

        # logging.info(query)
        # logging.info(text_response)

        hierarchy = []
        
        # split text into chunk of step_size and embed each chunk
        embedding_chunks, pivot_chunks = process_chunk(text_response, small_model.get_text_embedding)
        hierarchy.append({
            "layer": 0,
            "embedding_chunks": embedding_chunks,
            "pivot_chunks": pivot_chunks
        })

        for i in range(1, num_layers):
            embedding_chunks, pivot_chunks = process_chunk(embedding_chunks, pivot_chunks, average_embeddings)
            hierarchy.append({
                "layer": i,
                "embedding_chunks": embedding_chunks,
                "pivot_chunks": pivot_chunks
            })

        data.append({
            "item": item,
            "query": query,
            "text_response": text_response,
            "hierarchy": hierarchy
        })

    with open(os.path.join(experiment_path, "text_hierarchy_data.json"), "w") as f:
        json.dump({
            "data": data,
            "step_size": step_size,
            "num_layers": num_layers
        }, f, indent=4)

    logging.info("Done.")
