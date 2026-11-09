import torch
import torch.nn as nn


def pepe_crossentropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits - torch.amax(logits, dim=-1, keepdim=True).detach()
    return (torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True)) - logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)).mean()

@torch.no_grad()
def pepe_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.exp(pepe_crossentropy(logits, labels))

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



