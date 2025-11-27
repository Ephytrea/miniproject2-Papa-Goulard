import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

class CausalSelfAttn(nn.Module) :
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1) :
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x) :
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


class TransformerBlock(nn.Module) :
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1) :
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = CausalSelfAttn(embed_dim, num_heads, block_size, dropout)

        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x) :
        x = x + self.attn(self.ln1(x))
        out = x + self.mlp(self.ln2(x))
        return out


class Shakespeare(nn.Module) :
    def __init__(self, vocab_size, block_size, n_layers=12, embed_dim=768, num_heads=8, dropout=0.1) :
        super().__init__()

        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential( *[TransformerBlock(embed_dim, num_heads, block_size, dropout) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None) :
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None :
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0) :
        for _ in range(max_new_tokens) :
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
