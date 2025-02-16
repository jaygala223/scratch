import torch
import re
from gpt import SimpleTokenizer, GPTModel, PanchatantraDataset
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
import habana_frameworks.torch.hpu as hthpu

if torch.hpu.is_available():
    device = "hpu"
else:
    device = "cpu"

with open('panchatantra.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# result = re.findall(r'\w+|[^\w\s]', text)
preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# print(preprocessed_text[:10])
preprocessed_text = [item.strip() for item in preprocessed_text]
# print(preprocessed_text[:10])

all_words = sorted(list(set(preprocessed_text)))
all_words.extend(["<|end_of_text|>", "<|unk|>"])

vocab = {token: i for i, token in enumerate(all_words)}

tokenizer = SimpleTokenizer(vocab=vocab)
dataset = PanchatantraDataset(text=' '.join(preprocessed_text), tokenizer=tokenizer, context_length=8, stride=8)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

# print(dataset[0])

model_config = {
    "vocab_size": 1110,     # Vocabulary size
    "context_length": 16,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 16,           # Number of attention heads
    "n_layers": 16,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False 
}

learning_rate = 1e-3
num_epochs = 5

model = GPTModel(cfg=model_config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs, outputs = batch
        inputs.to(device)
        outputs.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
    
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).to(device), outputs.flatten())

        print("Loss:",loss.item())