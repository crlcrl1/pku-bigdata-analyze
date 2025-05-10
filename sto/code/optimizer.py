import torch
from torch import optim


class Adagrad(optim.Optimizer):

    def __init__(self, params, lr: float = 1e-2, eps: float = 1e-10):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, eps=eps)
        super(Adagrad, self).__init__(params, defaults)

        for param in self.param_groups:
            for p in param['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.zeros_like(p.data)
                state['lr'] = param['lr']
                state['eps'] = param['eps']

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                state['step'] += 1

                grad = p.grad.data
                state['sum'] += grad * grad
                std = state['sum'].add(state['eps']).sqrt()
                p.data.addcdiv_(grad, std, value=-group['lr'])

        return loss


class Adam(optim.Optimizer):

    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < betas[0] < 1.0):
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not (0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

        for param in self.param_groups:
            for p in param['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['lr'] = param['lr']
                state['betas'] = param['betas']
                state['eps'] = param['eps']

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                state['step'] += 1

                grad = p.grad.data
                beta1, beta2 = state['betas']
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = state['exp_avg_sq'].sqrt().add_(state['eps'])
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)

        return loss
