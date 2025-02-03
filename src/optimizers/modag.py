import torch
from typing import Callable, Optional
from torch.optim import Optimizer
from torch.optim.optimizer import required

class MoDAG(Optimizer):
    def __init__(self, params, lr=required, beta_ag=[0.9, 0.99], weight_decay=0, adap_factor=0.1):
        defaults = dict(lr=lr, beta_ag=beta_ag, weight_decay=weight_decay, adap_factor=adap_factor)
        super(MoDAG, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            total_mom = len(group['beta_ag'])
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Velocity terms from AggMo
                for beta in group['beta_ag']:
                    buf = self.state[p].get(f'velocity_{beta}', torch.zeros_like(p.data))
                    buf.mul_(beta).add_(d_p)
                    p.data.sub_(group['lr'] / total_mom , buf)
                    self.state[p][f'velocity_{beta}'] = buf

                # Update MADGRAD accumulation
                grad_acc = self.state[p].get('grad_accum', torch.zeros_like(p.data))
                grad_acc.add_(p.data, alpha=group['adap_factor'])
                self.state[p]['grad_accum'] = grad_acc

        return loss
