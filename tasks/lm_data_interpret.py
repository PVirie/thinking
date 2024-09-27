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
import torch

from utilities.utilities import *
from utilities.lm.openai_lm import Model as OpenAI_Model


# inputs = tokenizer("In order to build the advanced circuit in factorio, you must first", return_tensors="jax")

# for i in range(10):
#     outputs = model(**inputs)

#     # retrieve logts for next token
#     next_token_logits = outputs.logits[:, -1]

#     # decode outputs
#     next_token = next_token_logits.argmax(-1)

#     # update input_ids
#     inputs["input_ids"] = jnp.concatenate([inputs["input_ids"], jnp.expand_dims(next_token, axis=0)], axis=-1)
#     inputs["attention_mask"] = jnp.ones_like(inputs["input_ids"])

#     results = tokenizer.decode(inputs["input_ids"][0])
#     print(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # load llm data
    # build humn
    # train
    # save

    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "experiments", "lm_factorio")

    with open(os.path.join(experiment_path, "text_hierarchy_data.pkl"), "rb") as f:
        data = pickle.load(f)

    settings = data["settings"]

    sentence_format = "In order to build {} in factorio, here are the steps:"
    large_model = OpenAI_Model("gpt-4o")
    small_model = OpenAI_Model("gpt-4o-mini")

    # bootstrap embedding

    vocab_list = data["vocabulary"]["list"]
    vocab_embedding_tensor = torch.tensor(data["vocabulary"]["embeddings"])
    vocab_embedding_tensor = torch.reshape(torch.transpose(vocab_embedding_tensor, 0, 1), (1, vocab_embedding_tensor.shape[1], vocab_embedding_tensor.shape[0]))
    
    # generate and report

    reports = []
    item_data = data["data"]
    for item_datum in item_data:
        item = item_datum["item"]
        query = item_datum["query"]
        text_response = item_datum["text_response"]
        hierarchy = item_datum["hierarchy"]

        logging.info(f"Item: {item}")
        # logging.info(f"Original: {text_response}")

        lowest_layer = hierarchy[0]
        embedding_chunks = torch.tensor(lowest_layer["embedding_chunks"])
        pivot_chunks = lowest_layer["pivot_chunks"]

        # top n cosine similarity with the vocab
        n = 2
        cs = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        a = torch.reshape(embedding_chunks, (embedding_chunks.shape[0], embedding_chunks.shape[1], 1))
        metric = cs(a, vocab_embedding_tensor)
        top_n = torch.topk(metric, n, dim=1).indices

        content_draft = []
        for i in range(top_n.shape[0]):
            basis_texts = [vocab_list[top_n[i][j]] for j in range(n)]
            basis_texts = "\n".join(basis_texts)
            query = f"From these texts: \n {basis_texts} \n Here is the summary of the above texts in one sentence: "
            # now we generate the text
            basis_summary = small_model.get_chat_response(query, token_length=20)
            content_draft.append(basis_summary)


        draft = "\n".join(content_draft)
        query = f"Given the draft steps as follow: \n {draft} \n Here is the summary of the draft again step by step: "

        # now we generate the text
        result_text_response = small_model.get_chat_response(query, token_length=settings["max_text_length"])

        reports.append({
            "item": item,
            "query": query,
            "ground_truth_response": text_response,
            "result_response": result_text_response
        })


    with open(os.path.join(experiment_path, "text_response_report.json"), "w") as f:
        json.dump(reports, f, indent=4)

