import torch
import torch.nn as nn


def pepe_crossentropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits - torch.amax(logits, dim=-1, keepdim=True).detach()
    return (torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True)) - logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)).mean()

