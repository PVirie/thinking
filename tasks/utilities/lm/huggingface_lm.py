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
    
    def get_chat_response(session, query_message:str, token_length:int = 1000):
        inputs = tokenizer(query_message, return_tensors='pt')
        outputs = model.generate(inputs, max_new_tokens=token_length)
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return text


    @staticmethod
    def compute_mean_embedding(embeddings):
        # embeddings is a list of torch tensor
        return torch.mean(torch.stack(embeddings), dim=0)


    def get_text_embedding(self, text:str):
        embeddings = method_1(text)
        return embeddings


    def precompute_vocab_embedding_heuristic_score(self, target_embeddings):
        # get all token embeddings
        # dot product with target embeddings to get logits

        # target_embeddings has shape (embedding_size,)
        self.vocab_embedding = target_embeddings

        # get all token embeddings
        all_token_embeddings = model.get_input_embeddings().weight
        # all_token_embeddings has shape (vocab_size, embedding_size)

        # compute cosine similarity between all token embeddings and target embeddings
        logits = torch.matmul(all_token_embeddings, target_embeddings)
        self.vocab_embedding_heuristic_score = logits

    def reset_vocab_embedding_heuristic_score(self):
        self.vocab_embedding_heuristic_score = torch.zeros(model.config.vocab_size)


    def compute_look_ahead_score(self, text:str, num_future_slots=16):

        encoded_input = tokenizer(text, return_tensors='pt')

        # append future slot to encoded_input
        extended_input_ids = torch.cat(
            [encoded_input['input_ids'], 
             torch.zeros((1, num_future_slots), dtype=torch.long)], dim=-1
        )

        # apppend attention mask
        extended_attention_mask = torch.cat(
            [encoded_input['attention_mask'], 
             torch.zeros((1, num_future_slots), dtype=torch.long)], dim=-1
        )
        
        model_output = model(input_ids=extended_input_ids, attention_mask=extended_attention_mask, output_hidden_states=True)

        # sum embedding of the future slots
        future_slots_mean_embedding = model_output.hidden_states[-1][-num_future_slots:].mean(dim=1)
        future_slots_mean_embedding_error = torch.mean((self.vocab_embedding - future_slots_mean_embedding[0])**2)
 
        # embedding_hidden_states [batch, token_length, embedding_size]
        embedding_hidden_states = model_output.hidden_states[0]
        embedding_hidden_states.retain_grad()  # Ensure grads are retained
        
        # Backward pass
        future_slots_mean_embedding_error.backward()

        # The gradient will be with respect to the retained parts
        grad = embedding_hidden_states.grad[0, num_future_slots]

        print(grad)


    def get_best_next_token(self, text:str):
        # retrieve token score
        # multiply with heuristic score
        # get the best token

        encoded_input = tokenizer(text, return_tensors='pt')
        model_output = model(**encoded_input)
        logits = model_output.logits
        logits = logits[0, -1, :] + 0.5 * self.vocab_embedding_heuristic_score

        prob = torch.softmax(logits, dim=-1)
        best_token_id = torch.argmax(prob)
        best_token = tokenizer.decode(best_token_id)
        return best_token
    

    def sample_next_token(self, text:str, temperature=1.0, top_k=40):
        encoded_input = tokenizer(text, return_tensors='pt')
        model_output = model(**encoded_input)
        logits = model_output.logits
        logits = logits[0, -1, :] + 0.2 * self.vocab_embedding_heuristic_score
        
        # select only top_k, keep the rest as -inf
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, fill_value=-float('inf'))
        logits[indices] = values
        
        prob = torch.softmax(logits / temperature, dim=-1)

        next_token_id = torch.multinomial(prob, num_samples=1)
        next_token = tokenizer.decode(next_token_id)
        return next_token


if __name__ == "__main__":
    m = Model()
    print(m.get_chat_response("To start a business, this is a guideline:")) 
    embedding = m.get_text_embedding("In general, do research before taking action. Make sure that everything is planned and calculated.")
    print(embedding.shape)

    # print()
    # print("============ Research ===============")
    # print()

    # embedding = m.get_text_embedding("In general, do research before taking action. Make sure that everything is planned and calculated.")
    # m.precompute_vocab_embedding_heuristic_score(embedding)

    # m.compute_look_ahead_score("To start a business, this is a guideline:")

    # input = "To start a business, this is a guideline:"
    # print(input)
    # for i in range(40):
    #     next = m.sample_next_token(input)
    #     print(next, end="")
    #     input += next

    # with open(os.path.join("/app/log", f"research.txt"), "a") as f:
    #     f.write(input)


    # print()
    # print("============ Action ===============")
    # print()

    # embedding = m.get_text_embedding("Get to action first. Do not waste time on planning or research. Instead, learn from mistakes.")
    # m.precompute_vocab_embedding_heuristic_score(embedding)

    # input = "To start a business, this is a guideline:\n"
    # print(input)
    # for i in range(40):
    #     next = m.sample_next_token(input)
    #     print(next, end="")
    #     input += next

    # with open(os.path.join("/app/log", f"action.txt"), "a") as f:
    #     f.write(input)
