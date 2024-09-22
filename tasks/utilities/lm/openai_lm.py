import os
from openai import OpenAI

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
        return openai_session.embeddings.create(input = [text], model=model).data[0].embedding
    

if __name__ == "__main__":
    m = Model()

    print(m.get_chat_response("To start a business, this is a guideline:")) 
    embedding = m.get_text_embedding("In general, do research before taking action. Make sure that everything is planned and calculated.")
    # embedding is a list
    print(len(embedding))