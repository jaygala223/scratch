import torch
import time
import re
from gpt import SimpleTokenizer, GPTModel
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
import habana_frameworks.torch.hpu as hthpu
import matplotlib.pyplot as plt
from utils import generate_text_simple, generate_text_simple_with_kv_cache


if torch.hpu.is_available():
    device = "hpu"
else:
    device = "cpu"

with open('panchatantra.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # Remove all punctuation from the text
    text = re.sub(r'[^\w\s]', '', text)

preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)

preprocessed_text = [item.strip() for item in preprocessed_text]

all_words = sorted(list(set(preprocessed_text)))
all_words.extend(["<|end_of_text|>", "<|unk|>"])

vocab = {token: i for i, token in enumerate(all_words)}

tokenizer = SimpleTokenizer(vocab=vocab)


model_config = {
    "vocab_size": len(vocab),     # Vocabulary size
    "context_length": 999999,  # Context length
    "emb_dim": 1024,          # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 16,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False 
}


model = GPTModel(cfg=model_config)
# model.load_state_dict(torch.load('model_weights2.pth'))
model.to(device)

model.eval()

print("yay")

tps_simple_list = []
tps_kv_cache_list = []
max_new_tokens = 122

start_context = "एक बार एक".split()
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

start_time_simple = time.time()
out_simple, generation_times_per_token_simple = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=max_new_tokens,
    context_size=model_config["context_length"]
)
end_time_simple = time.time()

start_time_kv_cache = time.time()
out_kv_cache, generation_times_per_token_kv_cache = generate_text_simple_with_kv_cache(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=max_new_tokens,
    context_size=model_config["context_length"]
)
end_time_kv_cache = time.time()

time_simple = end_time_simple - start_time_simple
time_kv_cache = end_time_kv_cache - start_time_kv_cache

speedup_factor = time_simple / time_kv_cache

print(f"Time taken for simple generation: {time_simple} seconds")
print(f"Time taken for KV cache generation: {time_kv_cache} seconds")
print(f"Speedup factor: {speedup_factor}")
# plt.figure(figsize=(10, 6))
# plt.plot(generation_times_per_token_simple, label='Simple Generation')
# plt.plot(generation_times_per_token_kv_cache, label='KV Cache Generation')
# plt.xlabel('Token Generation Step')
# plt.ylabel('Time per Token (seconds)')
# plt.title('Token Generation Time Comparison')
# plt.legend()
# plt.savefig('generation_time_comparison.png')
