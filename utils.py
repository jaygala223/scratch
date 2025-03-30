### utility functions for llm training / inference
import torch

import time

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=0, top_k=None):
    # idx is (B, T) array of indices in the current context
    generation_times = []
    for _ in range(max_new_tokens):
        start_time = time.perf_counter()

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        end_time = time.perf_counter()
        generation_times.append(end_time - start_time)

    return idx, generation_times


def generate_text_simple_with_kv_cache(model, idx, max_new_tokens, context_size, temperature=0, top_k=None):
    
    generation_times = []

    #prefill
    start_time = time.perf_counter()
    logits = model(idx, use_kv_cache=True)
    logits = logits[:, -1, :]
    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    end_time = time.perf_counter()
    generation_times.append(end_time - start_time)


    pos_idx = idx.shape[1]

    idx = torch.cat((idx, idx_next), dim=1)

    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens-1):
        start_time = time.perf_counter()

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context

        with torch.no_grad():
            logits = model(idx_next, use_kv_cache=True, pos_idx=pos_idx)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
        pos_idx += 1
        end_time = time.perf_counter()
        generation_times.append(end_time - start_time)


    return idx, generation_times
