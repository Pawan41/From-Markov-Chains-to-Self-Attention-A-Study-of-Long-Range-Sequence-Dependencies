import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))

        mask = torch.tril(torch.ones(scores.shape[-2:])).to(x.device)
        scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=-1)
        return weights @ V, weights


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = SelfAttention(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, weights = self.attn(x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x, weights


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=2, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim),
            TransformerBlock(embed_dim)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)

        weights = None
        for block in self.blocks:
            x, weights = block(x)

        return self.fc(x), weights


