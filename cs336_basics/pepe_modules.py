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
        std = 2 / (in_features + out_features)
        self.w = nn.Parameter(nn.init.trunc_normal_(torch.ones(in_features, out_features),
                                                             mean=0.0,
                                                             std=std,
                                                             a=-3*std,
                                                             b=3*std))
        if dtype is not None:
            self.dtype = dtype
            self.w.type(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... d_in, d_in d_out -> ... d_out")


if __name__ == "__main__":
    linear = Linear(10, 7)
    inp = torch.randn(5, 10)
    out = linear(inp)
