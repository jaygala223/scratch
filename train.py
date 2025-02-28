import torch
import re
from tqdm import tqdm
from gpt import SimpleTokenizer, GPTModel, PanchatantraDataset
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
import habana_frameworks.torch.hpu as hthpu
import matplotlib.pyplot as plt

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
train_ratio = 0.8
test_ratio = 1 - train_ratio
train_size = int(len(preprocessed_text) * train_ratio)
train_data = preprocessed_text[:train_size]
test_data = preprocessed_text[train_size:]

train_dataset = PanchatantraDataset(input=train_data, tokenizer=tokenizer, context_length=8, stride=8)
test_dataset = PanchatantraDataset(input=test_data, tokenizer=tokenizer, context_length=8, stride=8)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False) #Added test dataloader

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

train_losses = []
test_losses = []

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

        if i%5 == 0:
            print(f"Loss at {i}: {loss.item()}")
    
    #Testing Logic
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            inputs, outputs = batch
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), outputs.flatten())
            test_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_dataloader)
    avg_test_loss = test_loss / len(test_dataloader)
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    print(f"Train Loss for epoch {epoch}: {avg_train_loss}")
    print(f"Test Loss for epoch {epoch}: {avg_test_loss}")
    model.train() #Switch back to training mode

#Plot the graph
epochs = range(num_epochs)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss vs Test Loss')
plt.legend()
plt.savefig('loss_plot.png')


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