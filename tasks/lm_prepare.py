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
import pickle

from utilities.utilities import *
from utilities.lm.huggingface_lm import Model as Small_Model
from utilities.lm.openai_lm import Model as Large_Model



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. clear existing data
    # 2. prepare parameters
    # 3. generate text
    # 4. tokenize text
    # 5. split into hierarchy
    # 6. embedding text
    # 7. save

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

    settings = {
        "text_chunk_size": 64,
        "step_size": 4,
        "num_layers": 3,
        "max_text_length": 2000
    }

    def average_embeddings(embedding_chunks):
        return Small_Model.compute_mean_embedding(embedding_chunks)

    def process_chunk(embedding_chunks, processor, step_size=4):
        # split chunks into sub_chunk of step_size
        abstract_chunks = [processor(embedding_chunks[i:i+step_size]) for i in range(0, len(embedding_chunks), step_size)]
        abstract_pivot_chunks = [[i, min(i + step_size, len(embedding_chunks))] for i in range(0, len(embedding_chunks), step_size)]
        return abstract_chunks, abstract_pivot_chunks
        
    def serialize_tensors(list_of_tensors):
        return [t.tolist() for t in list_of_tensors]

    data = []
    for item_i, item in enumerate(item_list):
        if item_i % 10 == 0:
            logging.info(f"Processing item {item_i} out of {len(item_list)}")

        query = sentence_format.format(item)
        text_response = large_model.get_chat_response(query, token_length=settings["max_text_length"])

        # logging.info(query)
        # logging.info(text_response)

        hierarchy = []
        
        # split text into chunk of step_size and embed each chunk
        embedding_chunks, pivot_chunks = process_chunk(text_response, small_model.get_text_embedding, step_size=settings["text_chunk_size"])
        hierarchy.append({
            "layer": 0,
            "embedding_chunks": serialize_tensors(embedding_chunks),
            "pivot_chunks": pivot_chunks
        })

        for i in range(1, settings["num_layers"]):
            embedding_chunks, pivot_chunks = process_chunk(embedding_chunks, average_embeddings, step_size=settings["step_size"])
            hierarchy.append({
                "layer": i,
                "embedding_chunks": serialize_tensors(embedding_chunks),
                "pivot_chunks": pivot_chunks
            })

        data.append({
            "item": item,
            "query": query,
            "text_response": text_response,
            "hierarchy": hierarchy
        })


    vocab_embeddings = []
    for item in item_list:
        vocab_embeddings.append(small_model.get_text_embedding(item).tolist())


    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")
    logging.info(f"Clearing the experiment directory: {experiment_path}")
    empty_directory(experiment_path)
    os.makedirs(experiment_path, exist_ok=True)

    with open(os.path.join(experiment_path, "text_hierarchy_data.pkl"), "wb") as f:
        pickle.dump({
            "data": data,
            "settings": settings,
            "vocabulary": {
                "embeddings": vocab_embeddings,
                "list": item_list
            }
        }, f)

    logging.info("Done.")
