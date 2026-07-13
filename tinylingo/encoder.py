import torch
import torch.nn as nn
from embeddings import TokenEmbedding
from multi_head_attention import MultiHeadAttention
from layers import PositionwiseFeedForward, ResidualConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.last_attn_weights = None

    def forward(self, x, mask=None):
        def self_attn_fn(x_):
            out, attn_weights = self.self_attn(x_, x_, x_, mask)
            self.last_attn_weights = attn_weights
            return out

        x = self.residual1(x, self_attn_fn)
        x = self.residual2(x, self.ffn)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1, embedding=None):
        super().__init__()
        self.embedding = embedding if embedding is not None else \
            TokenEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, token_ids, mask=None):
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


def create_padding_mask(token_ids, pad_id):
    """
    token_ids: (batch, seq_len)
    Returns: (batch, 1, seq_len) -- True at PAD key positions, broadcasts across
    all query positions (every query should ignore the same PAD keys equally).
    MultiHeadAttention.forward adds the heads dimension on top of this internally.
    """
    return (token_ids == pad_id).unsqueeze(1)


if __name__ == "__main__":
    import sys, os
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(THIS_DIR)
    sys.path.append(os.path.join(PROJECT_ROOT, "tokeniser"))

    from tokenizer_wrapper import TranslationTokenizer
    import pandas as pd

    tok = TranslationTokenizer(
        os.path.join(PROJECT_ROOT, "tokeniser", "artifacts", "tinylingo_spm.model"),
        max_seq_len=64
    )
    train_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "dataset", "processed", "tinylingo_train.parquet"))

    batch = train_df.sample(8, random_state=42)
    src_tensor = tok.batch_encode_source(batch["source_tagged"].tolist())

    d_model, num_heads, d_ff, num_layers = 256, 8, 1024, 6
    encoder = Encoder(
        vocab_size=tok.vocab_size, d_model=d_model, num_heads=num_heads,
        d_ff=d_ff, num_layers=num_layers, max_seq_len=64
    )

    mask = create_padding_mask(src_tensor, tok.pad_id)
    print("padding mask shape:", mask.shape)

    out = encoder(src_tensor, mask)
    print("encoder output shape:", out.shape)

    last_layer_attn = encoder.layers[-1].last_attn_weights
    print("last layer attention weights shape:", last_layer_attn.shape)

    pad_positions = (src_tensor[0] == tok.pad_id)
    if pad_positions.any():
        first_pad_idx = pad_positions.nonzero()[0].item()
        print(f"\nRow 0 first PAD position: {first_pad_idx}")
        print("Attention weight on that PAD key, head 0, query position 0:",
              last_layer_attn[0, 0, 0, first_pad_idx].item())
    else:
        print("\nRow 0 has no padding in this sample.")