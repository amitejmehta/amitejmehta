import torch


def get_last_token_logits(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # 0: for the first sequence in the batch, -1 for the last token in that sequence, : to exctract all the possible logits
    last_token_logits = logits[0, -1, :]

def get_top_k(tokenizer, last_token_logits, k):
    return [tokenizer.decode(i) for i in torch.topk(last_token_logits, k=k)]

    
def generate_token(inputs, model, sampling="greedy"):
    last_token_logits = get_last_token_logits(inputs, model)

    if sampling=="greedy":
        return last_token_logits.argmax()

def generate_token_w_caching(model, inputs):
    last_token_logits = get_last_token_logits(inputs, model)

    


if __name__ == "main":
