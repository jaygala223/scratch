import re
import torch.nn as nn
import torch
from torch.utils.data import Dataset

# tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.stoi = vocab # dict --> string: id
        self.itos = {i:s for s,i in vocab.items()}
    
    def encode(self, input: list):
        # preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = input
        preprocessed = [item.strip() for item in preprocessed]
        for item in preprocessed:
            if item in ['', ' ', '!', '"', '(', ')', ',', '.', '8', ':', ';', '?']:
                preprocessed.remove(item)
        preprocessed = [item if item in self.stoi else "<|unk|>" for item in preprocessed]
        ids = [self.stoi[item] for item in preprocessed]
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.itos[id] for id in ids])
        # text = re.sub(r'([,.:;?_!"()\']|--|\s)',r'\1',  text)
        return text


class PanchatantraDataset(Dataset):
    def __init__(self, input, tokenizer, context_length, stride=1):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(input)
        print(token_ids[:10])

        for i in range(0, len(token_ids) - context_length, stride):
            x = token_ids[i : i+context_length] # e.g [3, 41, 885, 696]
            y = token_ids[i+1: i+context_length+1] # e.g [41, 885, 696, 148]
            self.input_ids.append(torch.tensor(x))
            self.target_ids.append(torch.tensor(y))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_kv_cache=False):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        if use_kv_cache:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = keys
                self.v_cache = values
            else:
                self.k_cache = torch.cat((self.k_cache, keys), dim=1)
                self.v_cache = torch.cat((self.v_cache, values), dim=1)

            keys = self.k_cache
            values = self.v_cache

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, keys.shape[1], self.num_heads, self.head_dim)
        values = values.view(b, values.shape[1], self.num_heads, self.head_dim)
        queries = queries.view(b, queries.shape[1], self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        if not use_kv_cache or queries.shape[1] != 1: #indicating in kv cache mode but prefill
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        if use_kv_cache: context_vec = context_vec.contiguous().view(b, -1, self.d_out)
        else: 
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = context_vec[:, -1, :]
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_kv_cache=False):
        res = x
        x = self.norm1(x)
        x = self.att(x, use_kv_cache)   # Shape [batch_size, num_tokens, emb_size]
        # x = self.drop(x)
        x = x + res

        res = x
        x = self.norm2(x)
        x = self.ff(x)
        # x = self.drop(x)
        x = x + res

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_kv_cache = False, pos_idx=None):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        if not use_kv_cache:
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        elif use_kv_cache and pos_idx is None:
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        else:
            pos_embeds = self.pos_emb(torch.arange(pos_idx+1, device=in_idx.device))[-1:, :]
            x = tok_embeds + pos_embeds
        # x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, use_kv_cache)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
