"""
This script is used to prepare the data for the language model experiment.
I use large model to generate text data.
I use small model to tokenize and embed text data.

"""
import os
import logging
from typing import List, Any
import pickle
import torch

from utilities.utilities import *
from utilities.lm.openai_lm import Model as OpenAI_Model



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    large_model = OpenAI_Model("gpt-4o")
    small_model = OpenAI_Model("gpt-4o-mini")

    # 1. clear existing data
    # 2. prepare parameters
    # 3. generate text
    # 4. tokenize text
    # 5. split into hierarchy
    # 6. embedding text
    # 7. save

    #============================= Train data ===============================================

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

    start_sentence = "from scratch"
    goal_sentence_format = "building {} in factorio"
    prompt_format = "In order to achieve the goal of {} {}, here are the steps:"

    settings = {
        "prompt_format": prompt_format,
        "start_sentence": start_sentence,
        "goal_sentence_format": goal_sentence_format,
        "max_text_length": 2000
    }

    def split_and_embed_text(text, processor, separators=["\n"]):
        # split text into chunk of step_size and embed each chunk
        text_chunks = []
        embedding_chunks = []
        pivot_chunks = []

        start = 0
        for i, char in enumerate(text):
            if char in separators:
                end = i + 1
                if end - start > 1:
                    chunk = text[start:end].strip()
                    if len(chunk) == 0:
                        continue
                    text_chunks.append(chunk)
                    embedding_chunks.append(processor(chunk))
                    pivot_chunks.append([start, end])
                start = end
        return text_chunks, embedding_chunks, pivot_chunks


    def serialize_tensors(list_of_tensors):
        return [t.tolist() for t in list_of_tensors]

    vocab_embeddings = []
    vocab_list = []
    train_dataset = []
    for item_i, item in enumerate(item_list):
        if item_i % 10 == 0:
            logging.info(f"Processing item {item_i} out of {len(item_list)}")

        goal_sentence = goal_sentence_format.format(item)
        query = prompt_format.format(goal_sentence, start_sentence)
        text_response = large_model.get_chat_response(query, token_length=settings["max_text_length"])

        start_embedding = large_model.get_text_embedding(start_sentence).tolist()
        # goal_embedding = large_model.get_text_embedding(goal_sentence).tolist()
        goal_embedding = large_model.get_text_embedding(item).tolist()

        # logging.info(query)
        # logging.info(text_response)

        # split text into chunk of step_size and embed each chunk
        text_chunks, embedding_chunks, pivot_chunks = split_and_embed_text(text_response, large_model.get_text_embedding)

        vocab_embeddings.extend(embedding_chunks)
        vocab_list.extend(text_chunks)

        train_dataset.append({
            "item": item,
            "query": query,
            "text_response": text_response,
            "start_embedding": start_embedding,
            "goal_embedding": goal_embedding,
            "embedding_chunks": serialize_tensors(embedding_chunks),
            "pivot_chunks": pivot_chunks
        })

    #============================= Test data ===============================================

    item_list = [
        "Processing unit, advanced circuit, and electric circuit",
        "Utility science pack and Production science pack",
        "Stack inserter and express transport belt",
    ]

    test_dataset = []
    for item_i, item in enumerate(item_list):
        if item_i % 10 == 0:
            logging.info(f"Processing item {item_i} out of {len(item_list)}")

        goal_sentence = goal_sentence_format.format(item)
        query = prompt_format.format(goal_sentence, start_sentence)
        text_response = large_model.get_chat_response(query, token_length=settings["max_text_length"])

        start_embedding = large_model.get_text_embedding(start_sentence).tolist()
        goal_embedding = large_model.get_text_embedding(goal_sentence).tolist()

        # logging.info(query)

        test_dataset.append({
            "item": item,
            "query": query,
            "text_response": text_response,
            "start_embedding": start_embedding,
            "goal_embedding": goal_embedding,
        })


    #============================= Save ===============================================

    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")
    logging.info(f"Clearing the experiment directory: {experiment_path}")
    empty_directory(experiment_path)
    os.makedirs(experiment_path, exist_ok=True)

    with open(os.path.join(experiment_path, "text_hierarchy_data.pkl"), "wb") as f:
        pickle.dump({
            "train_set": train_dataset,
            "test_set": test_dataset,
            "settings": settings,
            "vocabulary": {
                "embeddings": serialize_tensors(vocab_embeddings),
                "list": vocab_list
            }
        }, f)

    logging.info("Done.")
