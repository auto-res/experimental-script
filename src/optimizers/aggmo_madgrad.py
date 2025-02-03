import torch
from typing import List, Optional, Callable

class AggMoMADGRAD(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-2, 
        betas: List[float] = [0.9, 0.7, 0.5], 
        weight_decay: float = 0.0, 
        eps: float = 1e-6
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super(AggMoMADGRAD, self).__init__(params, defaults)
        
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if 'sum_sq_grad' not in state:
                    state['sum_sq_grad'] = torch.zeros_like(p.data)
                    state['momentums'] = [torch.zeros_like(p.data) for _ in betas]
                sum_sq_grad = state['sum_sq_grad']
                momentums = state['momentums']
                sum_sq_grad.addcmul_(grad, grad, value=1)
                rms = sum_sq_grad.sqrt().add_(eps)
                w = grad.div(rms)
                for momentum, beta in zip(momentums, betas):
                    momentum.mul_(beta).add_(grad, alpha=lr)
                    w.add_(momentum, alpha=lr / len(betas))
                p.data.add_(-w)
        return loss
