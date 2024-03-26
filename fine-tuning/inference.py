import torch


def get_last_token_logits(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # 0: for the first sequence in the batch, -1 for the last token in that sequence, : to extract all the possible logits
    last_token_logits = logits[0, -1, :]


def get_top_k(inputs, model, tokenizer, k):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # 0: for the first sequence in the batch, -1 for the last token in that sequence, : to extract all the possible logits
    last_token_logits = logits[0, -1, :]
    return [tokenizer.decode(i) for i in torch.topk(last_token_logits, k=k).indices]


def generate_token(inputs, model, sampling="greedy"):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # 0: for the first sequence in the batch, -1 for the last token in that sequence, : to extract all the possible logits
    last_token_logits = logits[0, -1, :]

    if sampling == "greedy":
        next_token_id = last_token_logits.argmax()

    return next_token_id


def generate_token_w_caching(inputs, model, sampling="greedy"):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # 0: for the first sequence in the batch, -1 for the last token in that sequence, : to extract all the possible logits
    last_token_logits = logits[0, -1, :]

    if sampling == "greedy":
        next_token_id = last_token_logits.argmax()
    return next_token_id, outputs.past_key_values


def generate_no_caching(inputs, model, tokenizer, max_tokens, sampling="greedy"):
    generated_token_ids = []
    next_inputs = inputs
    for _ in range(max_tokens):
        next_token_id = generate_token(next_inputs, model, sampling)
        next_inputs = {
            "input_ids": torch.cat([next_inputs['input_ids'], next_token_id.reshape(1, 1)], dim=1),
            "attention_mask": torch.cat([next_inputs['attention_mask'], torch.tensor([[1]])], dim=1)
        }

        generated_token_ids.append(next_token_id)
    generated_text = tokenizer.decode(generated_token_ids)
    return generated_text


def generate(inputs, model, tokenizer, max_tokens, sampling="greedy"):
    generated_token_ids = []
    next_inputs = inputs
    for _ in range(max_tokens):
        next_token_id, past_key_values = generate_token_w_caching(
            next_inputs, model, sampling)
        next_inputs = {
            "input_ids": next_token_id.reshape(1, 1),
            "attention_mask": torch.cat([next_inputs['attention_mask'], torch.tensor([[1]])], dim=1),
            "past_key_values": past_key_values
        }
        generated_token_ids.append(next_token_id)

    generated_text = tokenizer.decode(generated_token_ids)
    return generated_text
