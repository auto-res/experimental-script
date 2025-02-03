import torch
from torch.optim.optimizer import Optimizer, required

class AdvancedOptimizer(Optimizer):
    def __init__(self, params, lr=required, betas=[0.0, 0.9, 0.99], momentum=0.9,
                 weight_decay=0, eps=1e-6):
        defaults = dict(lr=lr, betas=betas, momentum=momentum, 
                        weight_decay=weight_decay, eps=eps)
        super(AdvancedOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if 'momentum_buffers' not in state:
                    state['momentum_buffers'] = [torch.zeros_like(p.data) for _ in betas]
                    state['adaptive_avg'] = torch.zeros_like(p.data)

                buffers = state['momentum_buffers']
                adaptive_avg = state['adaptive_avg']

                for i, beta in enumerate(betas):
                    buffers[i].mul_(beta).add_(grad)

                if isinstance(eps, str):
                    eps = float(eps)
                
                avg_momentum = sum(buffers) / len(betas)
                adaptive_avg.add_(grad**2)
                rms = adaptive_avg.sqrt().add_(eps)
                p.data.addcdiv_(avg_momentum, rms, value=-lr)

        return loss
