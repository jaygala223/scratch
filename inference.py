import torch
import re
from tqdm import tqdm
from gpt import SimpleTokenizer, GPTModel, PanchatantraDataset
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
import habana_frameworks.torch.hpu as hthpu
import matplotlib.pyplot as plt
import time

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
    "context_length": 128,  # Context length
    "emb_dim": 1024,          # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 16,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False 
}


model = GPTModel(cfg=model_config)
model.load_state_dict(torch.load('model_weights_2.pth'))
model.to(device)

model.eval()

print("yay")
import matplotlib.pyplot as plt
import time
from utils import generate_text_simple, generate_text_simple_with_kv_cache



max_new_tokens_values = range(1, 32)
tps_simple_list = []
tps_kv_cache_list = []

for max_new_tokens in max_new_tokens_values:
    start_context = "एक बार एक".split()
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    start_time_simple = time.perf_counter()
    out_simple = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=max_new_tokens,
        context_size=model_config["context_length"]
    )
    end_time_simple = time.perf_counter()
    time_taken_simple = end_time_simple - start_time_simple
    tokens_per_second_simple = max_new_tokens / time_taken_simple
    tps_simple_list.append(tokens_per_second_simple)

    start_time_kv_cache = time.perf_counter()
    out_kv_cache = generate_text_simple_with_kv_cache(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=max_new_tokens,
        context_size=model_config["context_length"]
    )
    end_time_kv_cache = time.perf_counter()
    time_taken_kv_cache = end_time_kv_cache - start_time_kv_cache
    tokens_per_second_kv_cache = max_new_tokens / time_taken_kv_cache
    tps_kv_cache_list.append(tokens_per_second_kv_cache)

plt.plot(max_new_tokens_values, tps_simple_list, label='Without KV Cache')
plt.plot(max_new_tokens_values, tps_kv_cache_list, label='With KV Cache')
plt.xlabel('Max New Tokens')
plt.ylabel('Tokens per Second')
plt.title('Tokens per Second vs. Max New Tokens')
plt.legend()
plt.savefig('kv_cache_vs_no_kv_cache.png')
plt.show()
