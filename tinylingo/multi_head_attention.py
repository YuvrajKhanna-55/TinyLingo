import torch
import torch.nn as nn
from attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, q_input, k_input, v_input, mask=None):
        """
        Self-attention: call with q_input == k_input == v_input (same tensor).
        Cross-attention: q_input = decoder states, k_input = v_input = encoder output.

        mask: broadcastable to (batch, seq_len_q, seq_len_k), True = masked out.
        """
        batch_size = q_input.size(0)

        q = self.split_heads(self.q_proj(q_input), batch_size)
        k = self.split_heads(self.k_proj(k_input), batch_size)
        v = self.split_heads(self.v_proj(v_input), batch_size)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)
 

        combined = self.combine_heads(attn_out, batch_size)
        out = self.out_proj(combined)

        return out, attn_weights


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, d_model, num_heads = 2, 5, 256, 8

    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    out, attn_weights = mha(x, x, x)
    print("output shape:", out.shape)
    print("attention weights shape:", attn_weights.shape)
    print("attention row sums:", attn_weights.sum(dim=-1)[0, 0])


    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
    mask[:, :, -1] = True
    out_masked, attn_masked = mha(x, x, x, mask=mask)
    print("\nmasked (last) key weight across all heads:", attn_masked[:, :, :, -1])