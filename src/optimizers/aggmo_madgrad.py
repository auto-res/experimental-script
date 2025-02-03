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
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]
                if 'sum_sq_grad' not in state:
                    state['sum_sq_grad'] = torch.zeros_like(p.data)
                    state['momentums'] = [torch.zeros_like(p.data) for _ in betas]
                    state['step'] = 0
                
                state['step'] += 1
                sum_sq_grad = state['sum_sq_grad']
                momentums = state['momentums']
                
                # Update sum of squared gradients
                sum_sq_grad.addcmul_(grad, grad, value=1)
                
                # Compute adaptive learning rate
                rms = sum_sq_grad.sqrt().add_(eps)
                
                # Initialize update
                update = grad.div(rms)
                
                # Apply momentum
                for momentum, beta in zip(momentums, betas):
                    momentum.mul_(beta).add_(grad, alpha=1-beta)
                    update.add_(momentum.div(rms))
                
                # Apply update with learning rate
                p.data.add_(update, alpha=-lr)
        return loss
