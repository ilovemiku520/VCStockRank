# model/vcformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class VariableCentricAttention(nn.Module):
    """标准多头注意力（兼容所有形状）"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x):
        attn_output, attn_weights = self.attn(x, x, x)
        return attn_output, attn_weights


class VariableCentricTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(d_model)

        self.attn_layers = nn.ModuleList([
            VariableCentricAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = []

    def forward(self, x):
        x = self.pos_encoder(x)
        self.attention_weights = []

        for i in range(len(self.attn_layers)):
            attn_output, attn_weights = self.attn_layers[i](x)
            self.attention_weights.append(attn_weights)

            x = self.norm1[i](x + self.dropout(attn_output))
            ffn_output = self.ffn_layers[i](x)
            x = self.norm2[i](x + self.dropout(ffn_output))

        return x

    def get_attention_weights(self):
        return self.attention_weights