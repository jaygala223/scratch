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

# preprocessed_text = re.findall(r'\w+|[^\w\s]', text)
preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# print(preprocessed_text[:10])
preprocessed_text = [item.strip() for item in preprocessed_text]
# print(preprocessed_text[:10])

all_words = sorted(list(set(preprocessed_text)))
# punctuations = all_words[:19]
# all_words = all_words[19:] # remove punctuations from vocab
all_words.extend(["<|end_of_text|>", "<|unk|>"])

vocab = {token: i for i, token in enumerate(all_words)}

tokenizer = SimpleTokenizer(vocab=vocab)
train_ratio = 0.9
test_ratio = 1 - train_ratio
train_size = int(len(preprocessed_text) * train_ratio)
train_data = preprocessed_text[:train_size]
test_data = preprocessed_text[train_size:]

train_dataset = PanchatantraDataset(input=train_data, tokenizer=tokenizer, context_length=8, stride=8)
test_dataset = PanchatantraDataset(input=test_data, tokenizer=tokenizer, context_length=8, stride=8)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True) #Added test dataloader

model_config = {
    "vocab_size": len(vocab),     # Vocabulary size
    "context_length": 128,  # Context length
    "emb_dim": 1024,          # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 16,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False 
}

learning_rate = 1e-4
num_epochs = 1

model = GPTModel(cfg=model_config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

train_losses = []
test_losses = []


start = time.perf_counter()
for epoch in range(num_epochs):

    print("Epoch:", epoch)
    train_loss = 0

    for i,batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs, outputs = batch
        inputs.to(device)
        outputs.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
    
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).to(device), outputs.flatten())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # #Testing Logic
    # model.eval()
    # test_loss = 0
    # with torch.no_grad():
    #     for i, batch in enumerate(test_dataloader):
    #         inputs, outputs = batch
    #         inputs = inputs.to(device)
    #         outputs = outputs.to(device)
    #         logits = model(inputs)
    #         loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), outputs.flatten())
    #         test_loss += loss.item()
    
    # avg_train_loss = train_loss / len(train_dataloader)
    # avg_test_loss = test_loss / len(test_dataloader)
    # train_losses.append(avg_train_loss)
    # test_losses.append(avg_test_loss)
    # print(f"Train Loss for epoch {epoch}: {avg_train_loss}")
    # print(f"Test Loss for epoch {epoch}: {avg_test_loss}")
    # model.train() #Switch back to training mode

print(f"Total training duration: {time.perf_counter() - start}")

#Plot the graph
# epochs = range(num_epochs)
# plt.plot(epochs, train_losses, label='Train Loss')
# plt.plot(epochs, test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Train Loss vs Test Loss')
# plt.legend()
# plt.savefig('loss_plot.png')

model.eval()

print("yay")
import matplotlib.pyplot as plt
import time
from gpt import generate_text_simple, generate_text_simple_with_kv_cache



max_new_tokens_values = range(1, 128)
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
