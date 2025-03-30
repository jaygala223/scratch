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

learning_rate = 1e-5
num_epochs = 2

model = GPTModel(cfg=model_config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

train_losses = []
test_losses = []
start = time.perf_counter()

model.train()
for epoch in range(num_epochs):

    print("Epoch:", epoch)
    train_loss = 0

    for i,batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs, outputs = batch
        inputs = inputs.to(device)
        outputs = outputs.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
    
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).to(device), outputs.flatten())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        

print(f"Total training duration: {time.perf_counter() - start}")

torch.save(model.state_dict(), 'model_weights2.pth')