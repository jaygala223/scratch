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

# preprocessed_text = re.findall(r'\w+|[^\w\s]', text)
preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# print(preprocessed_text[:10])
preprocessed_text = [item.strip() for item in preprocessed_text]
# print(preprocessed_text[:10])

all_words = sorted(list(set(preprocessed_text)))
punctuations = all_words[:11]
all_words = all_words[11:] # remove punctuations from vocab
all_words.extend(["<|end_of_text|>", "<|unk|>"])

vocab = {token: i for i, token in enumerate(all_words)}

tokenizer = SimpleTokenizer(vocab=vocab)
dataset = PanchatantraDataset(input=preprocessed_text, tokenizer=tokenizer, context_length=8, stride=8)

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

learning_rate = 1e-4
num_epochs = 10

model = GPTModel(cfg=model_config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):

    print("Epoch:", epoch)

    for i,batch in enumerate(train_dataloader):
        inputs, outputs = batch
        inputs.to(device)
        outputs.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
    
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).to(device), outputs.flatten())

        # print("Loss:",loss.item())

        loss.backward()
        optimizer.step()

        if i%5 == 0:
            print(f"Loss at {i}: {loss.item()}")

print("yay")

from gpt import generate_text_simple

start_context = "एक बार एक".split()

encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
print("\nInput text:", start_context)
print("Encoded input text:", encoded)
print("encoded_tensor.shape:", encoded_tensor.shape)

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=10,
    context_size=model_config["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())

print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
print("\nOutput:", out)
print("Output length:", len(out[0]))
print("Output text:", decoded_text)