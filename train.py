import torch
import re
from gpt import SimpleTokenizer, GPTModel, PanchatantraDataset

with open('panchatantra.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# result = re.findall(r'\w+|[^\w\s]', text)
preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(preprocessed_text[:10])
preprocessed_text = [item.strip() for item in preprocessed_text]
print(preprocessed_text[:10])

all_words = sorted(list(set(preprocessed_text)))
all_words.extend(["<|end_of_text|>", "<|unk|>"])

vocab = {token: i for i, token in enumerate(all_words)}

tokenizer = SimpleTokenizer(vocab=vocab)
dataset = PanchatantraDataset(text=' '.join(preprocessed_text), tokenizer=tokenizer, context_length=8, stride=8)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

print(dataset[0])

model_config = {
    "vocab_size": 1110,     # Vocabulary size
    "context_length": 16,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 4,           # Number of attention heads
    "n_layers": 4,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False 
}

learning_rate = 1e-4
num_epochs = 1

model = GPTModel(cfg=model_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs, outputs = batch

        for i in range(len(inputs)):
            x = inputs[:i+1] # get the inputs
            y = outputs[i]
            logits = model(x)
            # print(logits.shape)
            predicted = torch.argmax(logits, dim=-1, keepdim=True)
            print(predicted.shape, y.shape)
            loss = loss_fn(predicted.to(torch.float32), y.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Loss:", loss.item())