import re

with open('panchatantra.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# result = re.findall(r'\w+|[^\w\s]', text)
preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed_text = [item.strip() for item in preprocessed_text]

all_words = sorted(list(set(preprocessed_text)))
all_words.extend(["<|end_of_text|>", "<|unk|>"])

vocab = {token: i for i, token in enumerate(all_words)}


# tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.stoi = vocab # dict --> string: id
        self.itos = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed]
        preprocessed = [item if item in self.stoi else "<|unk|>" for item in preprocessed]
        ids = [self.stoi[item] for item in preprocessed]
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.itos[id] for id in ids])
        text = re.sub(r'([,.:;?_!"()\']|--|\s)',r'\1',  text)
        return text

tokenizer = SimpleTokenizer(vocab=vocab)

print(tokenizer.encode("टोपी पहनकर"))

print(tokenizer.decode([436,0,631,1109]))


# dataloader
import torch
from torch.utils.data import Dataset, DataLoader

class PanchatantraDataset(Dataset):
    def __init__(self, text, tokenizer, context_length, stride=1):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(len(token_ids) - context_length, stride):
            x = token_ids[i : i+context_length] # e.g [3, 41, 885, 696]
            y = token_ids[i+1: i+context_length+1] # e.g [41, 885, 696, 148]
            self.input_ids.append(torch.tensor(x))
            self.target_ids.append(torch.tensor(y))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

