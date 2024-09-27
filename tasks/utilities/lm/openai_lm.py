import os
from openai import OpenAI
import torch

try:
    from . import base
except:
    from tasks.utilities.lm import base


class Model(base.Model):

    def __init__(self):
        self.openai_session = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30)
    
    def get_chat_response(self, query_message:str, token_length:int = 1000, system_prompt=None):
        messages = [
            {
                "role": "system",
                "content": f"Answer the following query using no more than {token_length} tokens. Be concise and do not add any introduction.",
            },
        ]
        if system_prompt is not None:
            messages.insert(0, {
                "role": "system",
                "content": system_prompt,
            })
        messages.append({
            "role": "user",
            "content": query_message,
        })
        response = self.openai_session.chat.completions.create(
            model = "gpt-4o",
            messages = messages,
            max_tokens = token_length,
        )
        return response.choices[0].message.content
    

    def get_text_embedding(self, text:str):
        model="text-embedding-3-small"
        text = text.replace("\n", " ")
        openai_results = self.openai_session.embeddings.create(input = [text], model=model).data[0].embedding
        return torch.tensor(openai_results)
    

if __name__ == "__main__":
    m = Model()

    print(m.get_chat_response("To start a business, this is a guideline:")) 
    embedding = m.get_text_embedding("In general, do research before taking action. Make sure that everything is planned and calculated.")
    print(embedding.shape)