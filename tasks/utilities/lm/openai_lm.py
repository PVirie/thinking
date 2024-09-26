import os
from openai import OpenAI
import torch

try:
    from . import base
except:
    from tasks.utilities.lm import base

openai_session = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Model(base.Model):
    
    def get_chat_response(self, query:str, token_length:int = 1000):
        response = openai_session.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Answer the following questions using no more than {token_length} tokens. Go to the steps directly do not add any introduction.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            max_tokens=token_length,
        )
        return response.choices[0].message.content
    

    def get_text_embedding(self, text:str):
        model="text-embedding-3-small"
        text = text.replace("\n", " ")
        return torch.tensor(openai_session.embeddings.create(input = [text], model=model).data[0].embedding)
    

if __name__ == "__main__":
    m = Model()

    print(m.get_chat_response("To start a business, this is a guideline:")) 
    embedding = m.get_text_embedding("In general, do research before taking action. Make sure that everything is planned and calculated.")
    print(embedding.shape)