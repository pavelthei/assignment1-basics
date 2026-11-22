import os
import torch
import typing
from typing import Iterable
import numpy.typing as npt

from math import cos, pi


def pepe_crossentropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits - torch.amax(logits, dim=-1, keepdim=True).detach()
    return (torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True)) - logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)).mean()

@torch.no_grad()
def pepe_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.exp(pepe_crossentropy(logits, labels))

@torch.no_grad()
def cosine_lr_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int):
    diff = cosine_cycle_iters - warmup_iters
    lr_diff = lr_max - lr_min
    if it < warmup_iters:
        return it * lr_max / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        coef = (it - warmup_iters) / diff
        return lr_min + (1 + cos(coef * pi)) * lr_diff/ 2
    return lr_min

def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> float:
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0

    # global L2 norm:  sqrt( Σ ‖gᵢ‖² )
    total_norm = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in params))

    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(scale)          # in-place, no extra alloc
        return total_norm.item()
    return total_norm.item()

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = torch.Tensor(dataset).to(device)
    inp = dataset[:-1]
    target = dataset[1:]

    samples_idx = torch.randint(0, dataset.shape[0] - context_length, (batch_size, ), device=device)
    samples_idx = samples_idx[:, None] + torch.arange(context_length, device=device)
    return inp[samples_idx], target[samples_idx]


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    iteration: int) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["iteration"]


class PepeAdamW(torch.optim.Optimizer):

    def __init__(self, params, 
                 lr: float,
                 betas: tuple[float, float],
                 eps: float, 
                 weight_decay: float, 
                 **kwargs):
        defaults = kwargs
        defaults.update({'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay})
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            lmd = group['weight_decay']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                # extraction
                g = p.grad.data
                m = self.state[p].get('m', torch.zeros_like(p))
                v = self.state[p].get('v', torch.zeros_like(p))
                t = self.state[p].get('t', 0) + 1

                # Param calculation
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Weight decay
                p.data = p.data * (1 - lr * lmd)

                # update the state
                self.state[p]['m'] = m
                self.state[p]['v'] = v
                self.state[p]['t'] = t

                p.data = p.data - lr * m_hat / (torch.sqrt(v_hat) + group['eps'])

        return loss



