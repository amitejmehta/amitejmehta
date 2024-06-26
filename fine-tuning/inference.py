import torch
import torch.nn.functional as F


def greedy_sampling(logits):
    return logits.argmax(dim=-1)


def top_k_sampling(probabilities, k=10):
    top_k_probs, top_k_ids = torch.topk(probabilities, k=k)
    top_k_probs = top_k_probs/torch.sum(top_k_probs)

    next_token_id = torch.multinomial(top_k_probs, num_samples=1)

    next_token_id = top_k_ids.gather(
        dim=-1, index=next_token_id)

    return next_token_id.flatten()


def top_p_sampling(probabilities, p=0.8):
    sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
    print(sorted_probs.shape)

    cum_sum = torch.cumsum(sorted_probs, dim=-1)
    top_p_idx = cum_sum <= p

    mask_idx = cum_sum.sum(dim=-1).unsqueeze(-1) - 1
    mask_idx = mask_idx.long()
    masked_probs = top_p_idx.scatter_(-1, mask_idx, 1)

    top_p_ids = sorted_ids[masked_probs]
    top_p_probs = sorted_probs[masked_probs]

    if top_p_probs.ndim == 1:
        top_p_probs = top_p_probs.unsqueeze(0)
        top_p_ids = top_p_ids.unsqueeze(0)

    top_p_probs = top_p_probs / torch.sum(top_p_probs, dim=-1)

    next_token_idx = torch.multinomial(top_p_probs, num_samples=1)

    next_token_id = top_p_ids.gather(
        dim=-1, index=next_token_idx)
    print(next_token_id)

    return next_token_id


def beam_search(probabilities, beam_width=2):
    top_probs, top_ids = torch.topk(probabilities, k=beam_width)
    print(top_ids.shape)
    return top_ids, top_probs


def generate_token(inputs, model, caching=True, sampling="greedy", temperature=1.0, k=10, p=0.8, beam_width=2):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # 0: for the first sequence in the batch, -1 for the last token in that sequence, : to extract all the possible logits
    # last_token_logits = logits[0, -1, :]
    # : for all seqeunces in the batch, -1 for the last token, : for all possible logits
    last_token_logits = logits[:, -1, :]

    if sampling == "greedy":
        # for a single sequence in the batch
        # next_token_id = last_token_logits.argmax()

        # for batched, we set dimension=-1 to take argmax over the column (possible logits)
        # rather than from the rows (sequences in the batch)
        next_token_ids = greedy_sampling(last_token_logits)

    last_token_logits = last_token_logits/temperature
    probabilities = F.softmax(last_token_logits, dim=-1)

    if sampling == "top_k":
        next_token_ids = top_k_sampling(probabilities, k=k)

    if sampling == "top_p":
        next_token_ids = top_p_sampling(probabilities, p=p)

    if sampling == "beam_search":
        next_token_ids, probabilities = beam_search(
            probabilities, beam_width=beam_width)
        if caching:
            return next_token_ids, probabilities, outputs.past_key_values
        else:
            return next_token_ids, probabilities

    if caching:
        return next_token_ids, outputs.past_key_values

    return next_token_ids


def get_top_k(input, model, tokenizer, sampling='greedy', i=0, k=10, temperature=1.0, p=0.2):

    for _ in range(i):
        next_token_id, past_key_values = generate_token(
            input, model, sampling=sampling, temperature=temperature, k=k)
        input = {
            "input_ids": next_token_id.reshape(1, 1),
            "attention_mask": torch.cat([input['attention_mask'], torch.tensor([[1]])], dim=1),
            "past_key_values": past_key_values
        }

    with torch.no_grad():
        outputs = model(**input)
    logits = outputs.logits

    # 0: for the first sequence in the batch, -1 for the last token in that sequence, : to extract all the possible logits
    last_token_logits = logits[0, -1, :]
    last_token_logits = last_token_logits/temperature
    probabilities = F.softmax(last_token_logits, dim=-1)
    top_k = torch.topk(last_token_logits, k=k)
    top_k_tokens = [tokenizer.decode(i) for i in top_k.indices]
    top_k_probabilities = [probabilities[i] for i in top_k.indices]
    return top_k_tokens, top_k_probabilities


def generate_one_sequence(input, model, tokenizer, max_tokens, caching=True, sampling="greedy", temperature=1.0, k=10, p=0.8):
    generated_token_ids = []
    next_inputs = input

    for _ in range(max_tokens):
        if caching:
            next_token_id, past_key_values = generate_token(
                next_inputs, model,
                sampling=sampling,
                temperature=temperature,
                k=k,
                p=p)
            next_inputs = {
                "input_ids": next_token_id.reshape(-1, 1),
                "attention_mask": torch.cat([next_inputs['attention_mask'], torch.tensor([[1]])], dim=1),
                "past_key_values": past_key_values
            }
        else:
            next_token_id = generate_token(
                next_inputs,
                model,
                caching=False,
                sampling=sampling,
                temperature=temperature,
                k=k,
                p=p)
            next_inputs = {
                "input_ids": torch.cat([next_inputs['input_ids'], next_token_id.reshape(1, 1)], dim=1),
                "attention_mask": torch.cat([next_inputs['attention_mask'], torch.tensor([[1]])], dim=1),
            }
        generated_token_ids.append(next_token_id)

    # we use batch_decode because generate_token's first dimension is batch_size
    generated_tokens = tokenizer.batch_decode(generated_token_ids)
    return generated_tokens


def generate(inputs, model, tokenizer, max_tokens, sampling="greedy", temperature=1.0, k=10, p=0.2):
    generated_token_ids = [[] for _ in range(inputs['input_ids'].shape[0])]

    attention_mask = inputs['attention_mask']
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    next_inputs = {
        "position_ids": position_ids,
        **inputs
    }

    for _ in range(max_tokens):
        next_token_ids, past_key_values = generate_token(
            next_inputs, model, sampling=sampling, temperature=temperature, k=k, p=0.2
        )

        next_inputs = {
            "input_ids": next_token_ids.reshape(-1, 1),
            "attention_mask": torch.cat([
                next_inputs["attention_mask"],
                torch.ones((next_token_ids.shape[0], 1)),
            ], dim=1),
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
            "past_key_values": past_key_values
        }

        [seq.append(next_token_ids[i])
         for i, seq in enumerate(generated_token_ids)]

    generated_tokens = tokenizer.batch_decode(
        generated_token_ids)

    return ["".join(tokens) for tokens in generated_tokens]


def gen_w_beam_search(inputs, model, tokenizer, max_tokens, sampling="greedy", temperature=1.0, k=10, p=0.2, beam_width=2):
    generated_token_ids = [[] for _ in range(inputs['input_ids'].shape[0])]

    attention_mask = inputs['attention_mask']
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    next_inputs = {
        "position_ids": position_ids,
        **inputs
    }

    next_token_ids, probs, past_key_values = generate_token(
        next_inputs, model, sampling=sampling, temperature=temperature, k=k, p=0.2, beam_width=beam_width)
    for i in range(1, max_tokens):
        beams = [[] for _ in range(beam_width)]
        for j in range(beam_width):
            next_token_ids[:, j], probs[:, j], past_key_values = generate_token(
                next_inputs[:, j], model, sampling=sampling, temperature=temperature, k=k, p=0.2, beam_width=beam_width)
            beams.append((next_token_ids, probs, past_key_values))

        next_inputs = {
            "input_ids": next_token_ids.reshape(-1, 1),
            "attention_mask": torch.cat([
                next_inputs["attention_mask"],
                torch.ones((next_token_ids.shape[0], 1)),
            ], dim=1),
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
            "past_key_values": past_key_values
        }

        [seq.append(next_token_ids[i])
         for i, seq in enumerate(generated_token_ids)]

    generated_tokens = tokenizer.batch_decode(
        generated_token_ids)

    return ["".join(tokens) for tokens in generated_tokens]
