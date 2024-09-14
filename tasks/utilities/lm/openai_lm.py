import os
from openai import OpenAI

try:
    from . import base
except:
    from tasks.utilities.lm import base

openai_session = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Model(base.Model):
    
    def get_chat_response(self, query:str):
        response = openai_session.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a factorio expert to help guiding me create a factory. Please tell me how to get what I want step by step. No description, just steps. Make sure that each step is separated by a line break.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    

    def get_text_embedding(self, text:str):
        model="text-embedding-3-small"
        text = text.replace("\n", " ")
        return openai_session.embeddings.create(input = [text], model=model).data[0].embedding
    

if __name__ == "__main__":
    m = Model()

    print(m.get_chat_response("How to build advanced circuits?"))
    embedding = m.get_text_embedding("advanced circuits")
    # embedding is a list
    print(len(embedding))