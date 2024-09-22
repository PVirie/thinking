
class Model:

    def get_chat_response(query:str, token_length:int = 0):
        raise NotImplementedError()
    

    def get_text_embedding(text:str):
        raise NotImplementedError()