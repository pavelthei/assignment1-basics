import math
import torch
import torch.nn as nn

from einops import einsum

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


if __name__ == "__main__":
    norm_layer = SwiGLU(192)
    ids = torch.randn((5, 192))
    out = norm_layer(ids)
    print(out.shape)
