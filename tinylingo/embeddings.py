import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        assert d_model % 2 == 0, "PositionalEncoding requires d_model to be even for sin/cos encoding."

        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)

        sin_vals = torch.sin(position / div_term)   
        cos_vals = torch.cos(position / div_term)   

        pe = torch.stack([sin_vals, cos_vals], dim=2)      
        pe = pe.flatten(start_dim=1, end_dim=2)             

        self.register_buffer('pe', pe)   

    def forward(self, seq_len):
        return self.pe[:seq_len, :]   


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)  

        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.size()

        embeddings = self.embedding(token_ids)             
        embeddings = embeddings * math.sqrt(self.d_model)   
        pos = self.positional_encoding(seq_len)               

        out = self.dropout(embeddings + pos)   
        return out                              