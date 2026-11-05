import math
import torch
import torch.nn as nn

from einops import einsum, rearrange


class Softmax(nn.Module):

    def __init__(self) -> None:
        super(Softmax, self).__init__()

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        x = x - torch.amax(x, dim=i, keepdim=True)
        return torch.exp(x) / torch.sum(torch.exp(x), dim=i, keepdim=True)


class Linear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super(Linear, self).__init__()
        self.device = device
        std = math.sqrt(2 / (in_features + out_features))
        self.w = nn.Parameter(nn.init.trunc_normal_(torch.ones(in_features, out_features),
                                                             mean=0.0,
                                                             std=std,
                                                             a=-3*std,
                                                             b=3*std))
        if dtype is not None:
            self.dtype = dtype
            self.w = self.w.to(dtype)

        if self.device:
            self.w = self.w.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... d_in, d_in d_out -> ... d_out")


class Embedding(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embeddings_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(Embedding, self).__init__()
        self.device = device

        self.embeddings = nn.Parameter(nn.init.trunc_normal_(torch.ones(num_embeddings, embeddings_dim),
                                                             mean=0.0,
                                                             std=1.0,
                                                             a=-3.0,
                                                             b=3.0))

        if dtype is not None:
            self.dtype = dtype
            self.embeddings = self.embeddings.to(dtype)

        if self.device:
            self.embeddings = self.embeddings.to(self.device)

    def forward(self, embed_ids: torch.LongTensor):
        return self.embeddings[embed_ids]


class RMSNorm(nn.Module):

    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model))
        if dtype is not None:
            self.dtype = dtype
            self.g = self.g.to(dtype)

        if self.device:
            self.g = self.g.to(self.device)

    def forward(self, x: torch.FloatTensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x = (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))
        x = x * self.g
        return x.to(in_dtype)


class SwiGLU(nn.Module):

    def __init__(self,
                 d_model: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        d_ff = 8 * d_model // 3
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    @staticmethod
    def swish(x: torch.FloatTensor):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.FloatTensor):
        x1 = self.w1(x)
        x2 = self.w3(x)
        x = self.swish(x1) * x2
        x = self.w2(x)
        return x


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta_k = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        thetas = torch.outer(pos, theta_k).repeat_interleave(2, dim=-1)  # T x d
        self.register_buffer("sins", thetas.sin(), persistent=False)
        self.register_buffer("coss", thetas.cos(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        xx = torch.stack([-x2, x1], dim=-1).flatten(-2)  # [-x1, x0, -x3, x2, ...]
        return x * self.coss[token_positions] + xx * self.sins[token_positions]


class ScaledDotProduct(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.softmax = Softmax()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.BoolTensor | None = None) -> torch.Tensor:
        logits = einsum(q, k, "... seq_queries d_k, ... seq_keys d_k -> ... seq_queries seq_keys") / math.sqrt(self.d_k)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        weights = self.softmax(logits, -1)
        return einsum(weights, v, "... seq_queries seq_keys, ... seq_keys d_v -> ... seq_queries d_v")


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 rope: RoPE | None = None):
        super().__init__()
        self.d_model = d_model
        self.w_q, self.w_k, self.w_v, self.w_o = (
            nn.Parameter(torch.empty(d_model, d_model)) for _ in range(4)
        )
        std = (1 / d_model) ** 0.5  # sqrt(2 / (d_model + d_model))
        for w in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.se = ScaledDotProduct(self.d_k)
        self.rope = rope

    def forward(self,
                x: torch.Tensor,
                token_positions: torch.LongTensor | None = None) -> torch.Tensor:
        projection_operation = "... seq i, o i -> ... seq o"
        seq_len = x.shape[-2]
        q = einsum(x, self.w_q, projection_operation)
        k = einsum(x, self.w_k, projection_operation)
        v = einsum(x, self.w_v, projection_operation)

        q, k, v = (
            rearrange(m, "... seq (h d) -> ... h seq d", h=self.num_heads)
            for m in (q, k, v)
        )
        if self.rope and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn = self.se(q, k, v, mask)
        attn = rearrange(attn, "... h seq d -> ... seq (h d)")
        return einsum(attn, self.w_o, "... seq i, o i -> ... seq o")



if __name__ == "__main__":
    rope = RoPE(10000, 128, 1024)
    in_features = torch.randn((2, 10, 128))
    out = rope(in_features, torch.randint(0, 128, (2, 10)))
