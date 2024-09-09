import contextlib
import logging
import argparse
import json
from pydantic import BaseModel
from openai import OpenAI
import torch
import os
os.environ['HF_HOME'] = '/app/cache/'

from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# use gpt-4o to generate text data
# use selected lm to token and embed text data

from utilities import *
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import core


def get_chat_response(session, query_message:str):
    response = session.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a factorio expert to help guiding me create a factory. Please tell me how to get what I want step by step. No description, just steps. Make sure that each step is separated by a line break.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query_message},
                ],
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def get_text_embedding_with_lm(text:str):
    # using average pooling to get the embedding from the last hidden state
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state.mean(dim=1).squeeze().tolist()


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
        openai_session = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
