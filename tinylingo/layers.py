import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class ResidualConnection(nn.Module):
    """
    Wraps a sublayer (attention or FFN) with: LayerNorm(x + Dropout(Sublayer(x)))
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_fn):
        return self.norm(x + self.dropout(sublayer_fn(x)))


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model, d_ff = 2, 5, 256, 1024

    x = torch.randn(batch, seq_len, d_model) * 10 + 5  

    ln = LayerNormalization(d_model)
    out = ln(x)
    print("LayerNorm output shape:", out.shape)
    print("Per-position mean:", out.mean(dim=-1)[0, 0].item())
    print("Per-position std:", out.std(dim=-1)[0, 0].item())

    ffn = PositionwiseFeedForward(d_model, d_ff)
    ffn_out = ffn(x)
    print("\nFFN output shape:", ffn_out.shape)

    residual = ResidualConnection(d_model)
    res_out = residual(x, ffn)
    print("\nResidual+Norm wrapping FFN output shape:", res_out.shape)
    print("Per-position mean after residual+norm:", res_out.mean(dim=-1)[0, 0].item())