import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: (..., seq_len_q, d_k)
    k: (..., seq_len_k, d_k)
    v: (..., seq_len_k, d_v)
    mask: optional, broadcastable to (..., seq_len_q, seq_len_k), True = mask out
          (leading dims written as '...' since this same function will later
          be reused for multi-head attention, which adds a heads dimension)

    Returns:
      output: (..., seq_len_q, d_v)
      attention_weights: (..., seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9) 

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_k, d_v = 2, 5, 8, 8

    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_v)

    out, attn = scaled_dot_product_attention(q, k, v)
    print("output shape:", out.shape)
    print("attention shape:", attn.shape)
    print("attention row sums:", attn.sum(dim=-1))

    mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
    mask[:, :, -1] = True
    out_masked, attn_masked = scaled_dot_product_attention(q, k, v, mask=mask)
    print("\nattention weight on masked (last) key position:", attn_masked[:, :, -1])