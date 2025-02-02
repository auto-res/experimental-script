from typing import List, Dict, Any, Optional, Callable
import torch
from torch.optim import Optimizer

class AMADGRADOptimizer(Optimizer):
    def __init__(self, params, lr: float = 0.01, beta_values: List[float] = [0.9, 0.99, 0.999], eps: float = 1e-8):
        defaults = dict(lr=lr, beta_values=beta_values, eps=eps)
        super().__init__(params, defaults)
        
        for group in self.param_groups:
            group['step_counter'] = 0
            group['velocities'] = [{} for _ in beta_values]
            
            for p in group['params']:
                state = self.state[p]
                for i in range(len(beta_values)):
                    state[f'velocity_{i}'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group['step_counter'] += 1
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                for i, beta in enumerate(group['beta_values']):
                    vel = state[f'velocity_{i}']
                    vel.mul_(beta).add_(grad, alpha=1-beta)
                
                velocities = torch.stack([state[f'velocity_{i}'] for i in range(len(group['beta_values']))])
                avg_velocity = torch.mean(velocities, dim=0)
                
                denom = torch.tensor((group['step_counter'] + 1) ** 2, device=p.device)
                p.add_(avg_velocity.div(denom.sqrt().add(group['eps'])), alpha=-group['lr'])

        return loss
