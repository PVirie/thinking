
class Model:

    def get_chat_response(self, query_message:str, token_length:int = 0, system_prompt=None):
        raise NotImplementedError()
    

    def get_text_embedding(self, text:str):
        raise NotImplementedError()