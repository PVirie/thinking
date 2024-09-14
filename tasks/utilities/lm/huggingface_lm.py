import os
import torch

try:
    from . import base
except:
    from tasks.utilities.lm import base

os.environ['HF_HOME'] = '/app/cache/'

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

def method_1(text):
    # https://discuss.huggingface.co/t/get-word-embeddings-from-transformer-model/6929/2
    encoded_input = tokenizer(text, return_tensors='pt')
    model_output = model(**encoded_input, output_hidden_states=True)
    attention_mask = encoded_input['attention_mask']

    token_embeddings = model_output.hidden_states[-1]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    average_embeddings = sum_embeddings / sum_mask

    # flatten to a list of floats
    return average_embeddings.flatten()


def method_2(text):
    # https://stackoverflow.com/questions/76051807/automodelforcausallm-for-extracting-text-embeddings
    # https://stackoverflow.com/questions/76926025/sentence-embeddings-from-llama-2-huggingface-opensource
    encoded_input = tokenizer(text, return_tensors='pt')
    model_output = model(**encoded_input, output_hidden_states=True)
    attention_mask = encoded_input['attention_mask']

    last_hidden_state = model_output.hidden_states[-1]
    weights_for_non_padding = attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

    return sentence_embeddings.flatten()


# This does not work, it returns a tensor of shape (1, num_tokens, vocab_size)
# embedding_pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, return_tensors='pt')
# def method_3(text):
#     # https://discuss.huggingface.co/t/extracting-token-embeddings-from-pretrained-language-models/6834/9

#     data = embedding_pipe(text, output_hidden_states=True)
#     # data is 3 dimensions of shape (1, num_tokens, vocab_size)
#     # weight average the embeddings
#     embeddings = data[0].mean(dim=0)
#     return embeddings.flatten()


class Model(base.Model):
    
    def get_chat_response(session, query_message:str):
        messages = [
            {
                "role": "system",
                "content": "You are a factorio expert to help guiding me create a factory. Please tell me how to get what I want step by step. No description, just steps. Make sure that each step is separated by a line break.",
            },
            {
                "role": "user",
                "content": query_message
            }
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        outputs = model.generate(inputs, max_new_tokens=1000)
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return text


    def get_text_embedding(self, text:str):
        return method_1(text)
        

    def precompute_vocab_embedding_heuristic_score(self, target_embeddings):
        # get all token embeddings
        # dot product with target embeddings to get logits

        # target_embeddings has shape (embedding_size,)

        # get all token embeddings
        all_token_embeddings = model.get_input_embeddings().weight
        # all_token_embeddings has shape (vocab_size, embedding_size)

        # compute cosine similarity between all token embeddings and target embeddings
        logits = torch.matmul(all_token_embeddings, target_embeddings)
        self.vocab_embedding_heuristic_score = logits


    def get_best_next_token(self, text:str):
        # retrieve token score
        # multiply with heuristic score
        # get the best token

        encoded_input = tokenizer(text, return_tensors='pt')
        model_output = model(**encoded_input)


        logits = model_output.logits
        logits = logits[0, -1, :]
        prob = torch.softmax(logits, dim=-1)
        scores = prob * self.vocab_embedding_heuristic_score
        best_token_id = torch.argmax(scores)
        best_token = tokenizer.decode(best_token_id)
        return best_token
    

if __name__ == "__main__":
    m = Model()

    # print(m.get_chat_response("How to build advanced circuits?"))
    embedding = m.get_text_embedding("Check recipe first")

    m.precompute_vocab_embedding_heuristic_score(embedding)

    input = "Starting from scratch, The steps to build a factory that can produce science pack 3 are:\n"
    for i in range(30):
        next = m.get_best_next_token(input)
        input += next

    print(input)