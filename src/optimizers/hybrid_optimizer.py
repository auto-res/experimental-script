from typing import List, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer

class HybridOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: List[float] = [0.0, 0.9, 0.99],
        weight_decay: float = 0,
        eps: float = 1e-6
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            betas = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = {}
                    state['sum_grad_buffer'] = torch.zeros_like(grad)
                    for beta in betas:
                        state['momentum_buffer'][beta] = torch.zeros_like(grad)

                for beta in betas:
                    buf = state['momentum_buffer'][beta]
                    buf.mul_(beta).add_(grad)
                    p.add_(buf, alpha=-group['lr'] / len(betas))

                sum_grad = state['sum_grad_buffer']
                sum_grad.add_(grad.square())
                rms_grad = sum_grad.add(eps).sqrt()

                p.addcdiv_(grad, rms_grad, value=-group['lr'])

        return loss
