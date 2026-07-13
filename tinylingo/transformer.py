import torch
import torch.nn as nn
from embeddings import TokenEmbedding
from encoder import Encoder, create_padding_mask
from decoder import Decoder
from masks import create_decoder_self_attention_mask, create_cross_attention_mask


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_encoder_layers, num_decoder_layers, max_seq_len,
                 dropout=0.1, tie_weights=True):
        super().__init__()

        shared_embedding = TokenEmbedding(vocab_size, d_model, max_seq_len, dropout) if tie_weights else None

        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_encoder_layers,
                                max_seq_len, dropout, embedding=shared_embedding)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_decoder_layers,
                                max_seq_len, dropout, embedding=shared_embedding)

        self.output_proj = nn.Linear(d_model, vocab_size)
        if tie_weights:
            self.output_proj.weight = self.decoder.embedding.embedding.weight

    def forward(self, src_token_ids, decoder_input_ids, pad_id):
        enc_mask = create_padding_mask(src_token_ids, pad_id)
        encoder_output = self.encoder(src_token_ids, enc_mask)

        dec_self_mask = create_decoder_self_attention_mask(decoder_input_ids, pad_id)
        cross_mask = create_cross_attention_mask(src_token_ids, pad_id)

        decoder_output = self.decoder(decoder_input_ids, encoder_output, dec_self_mask, cross_mask)
        logits = self.output_proj(decoder_output)
        return logits