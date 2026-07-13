import torch
import torch.nn as nn
from embeddings import TokenEmbedding
from multi_head_attention import MultiHeadAttention
from layers import PositionwiseFeedForward, ResidualConnection


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.residual1 = ResidualConnection(d_model, dropout)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.residual2 = ResidualConnection(d_model, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

        self.last_self_attn_weights = None
        self.last_cross_attn_weights = None

    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        def self_attn_fn(x_):
            out, attn_w = self.self_attn(x_, x_, x_, self_attn_mask)
            self.last_self_attn_weights = attn_w
            return out
        x = self.residual1(x, self_attn_fn)

        def cross_attn_fn(x_):
            out, attn_w = self.cross_attn(x_, encoder_output, encoder_output, cross_attn_mask)
            self.last_cross_attn_weights = attn_w
            return out
        x = self.residual2(x, cross_attn_fn)

        x = self.residual3(x, self.ffn)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len,
                 dropout=0.1, embedding=None):
        super().__init__()
        self.embedding = embedding if embedding is not None else \
            TokenEmbedding(vocab_size, d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, target_token_ids, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        x = self.embedding(target_token_ids)
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        return x


if __name__ == "__main__":
    import sys, os
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(THIS_DIR)
    sys.path.append(os.path.join(PROJECT_ROOT, "tokeniser"))

    from tokenizer_wrapper import TranslationTokenizer
    from encoder import Encoder, create_padding_mask
    from masks import create_decoder_self_attention_mask, create_cross_attention_mask
    import pandas as pd

    tok = TranslationTokenizer(
        os.path.join(PROJECT_ROOT, "tokeniser", "artifacts", "tinylingo_spm.model"),
        max_seq_len=64
    )
    train_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "dataset", "processed", "tinylingo_train.parquet"))

    batch = train_df.sample(8, random_state=42)
    src_tensor = tok.batch_encode_source(batch["source_tagged"].tolist())
    tgt_tensor = tok.batch_encode_target(batch["target"].tolist())

    d_model, num_heads, d_ff, num_layers = 256, 8, 1024, 6

    encoder = Encoder(tok.vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=64)
    decoder = Decoder(tok.vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=64)

    enc_mask = create_padding_mask(src_tensor, tok.pad_id)
    encoder_output = encoder(src_tensor, enc_mask)
    print("encoder output shape:", encoder_output.shape)

    dec_self_mask = create_decoder_self_attention_mask(tgt_tensor, tok.pad_id)
    cross_mask = create_cross_attention_mask(src_tensor, tok.pad_id)

    decoder_output = decoder(tgt_tensor, encoder_output, dec_self_mask, cross_mask)
    print("decoder output shape (expect 8, 64, 256):", decoder_output.shape)

    last_layer = decoder.layers[-1]
    print("\nself-attn weights shape:", last_layer.last_self_attn_weights.shape)
    print("cross-attn weights shape:",
          last_layer.last_cross_attn_weights.shape)

    src_pad_positions = (src_tensor[0] == tok.pad_id).nonzero().flatten()
    if len(src_pad_positions) > 0:
        p = src_pad_positions[0].item()
        print(f"\nSource PAD at position {p}. Cross-attn weight from decoder query 0, head 0:",
              last_layer.last_cross_attn_weights[0, 0, 0, p].item())