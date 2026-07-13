import torch


def create_padding_mask(token_ids, pad_id):
    """
    token_ids: (batch, seq_len)
    Returns: (batch, 1, seq_len) -- True at PAD key positions.
    Reusable for: encoder self-attention, decoder cross-attention (source-side PAD),
    and as one component of the decoder self-attention mask (target-side PAD).
    """
    return (token_ids == pad_id).unsqueeze(1)


def create_causal_mask(seq_len, device=None):
    """
    Returns: (seq_len, seq_len) -- True where key position j is strictly
    in the future relative to query position i (upper triangular, excluding diagonal).
    Used only in decoder self-attention.
    """
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def create_decoder_self_attention_mask(token_ids, pad_id):
    """
    token_ids: (batch, seq_len) -- the DECODER's own input sequence
    Returns: (batch, seq_len, seq_len) -- combined causal + padding mask.
    """
    batch_size, seq_len = token_ids.shape
    causal = create_causal_mask(seq_len, device=token_ids.device).unsqueeze(0)   
    padding = (token_ids == pad_id).unsqueeze(1)                                
    return causal | padding


def create_cross_attention_mask(src_token_ids, pad_id):
    """
    Decoder cross-attention: query positions come from the decoder, but the mask
    only needs to hide SOURCE-side padding -- identical to the encoder's own mask.
    Kept as a separate named function for clarity at the call site in Phase 8,
    even though the logic is the same as create_padding_mask.
    """
    return create_padding_mask(src_token_ids, pad_id)


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
    tgt_tensor = tok.batch_encode_target(batch["target"].tolist())

    dec_self_mask = create_decoder_self_attention_mask(tgt_tensor, tok.pad_id)
    cross_mask = create_cross_attention_mask(src_tensor, tok.pad_id)

    print("decoder self-attention mask shape:", dec_self_mask.shape)
    print("cross-attention mask shape:", cross_mask.shape)

    row = 0
    print(f"\nRow {row} decoder self-mask, query position 0:")
    print("  masked positions:", dec_self_mask[row, 0].nonzero().flatten().tolist()[:10], "...")
    print("  key position 0 masked?", dec_self_mask[row, 0, 0].item())
    print("  key position 1 masked?", dec_self_mask[row, 0, 1].item())

    pad_positions = (tgt_tensor[row] == tok.pad_id).nonzero().flatten()
    if len(pad_positions) > 0:
        first_pad = pad_positions[0].item()
        print(f"\nFirst PAD position in target row {row}: {first_pad}")
        print(f"  masked for the LAST query position (should be True):",
              dec_self_mask[row, -1, first_pad].item())