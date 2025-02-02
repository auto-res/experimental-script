import torch
from torch.optim import Optimizer
from typing import Dict, List, Optional, Callable

class CustomOptimizer(Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        betas: List[float] = [0.0, 0.99], 
        weight_decay: float = 0, 
        adaptive_rate: float = 1.0
    ):
        defaults = dict(
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay, 
            adaptive_rate=adaptive_rate
        )
        super(CustomOptimizer, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta = group['betas'][0]
            adaptive_rate = group['adaptive_rate']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad.data
                
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = state['momentum_buffer']
                buf.mul_(beta).add_(d_p)
                
                ad_grad = buf / (torch.norm(buf, 2) + adaptive_rate)
                
                p.data.add_(ad_grad, alpha=-group['lr'])
        
        return loss
