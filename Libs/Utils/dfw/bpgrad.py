import torch
import torch.optim as optim

from torch.optim.optimizer import required


class BPGrad(optim.Optimizer):
    r"""
    Implements BPGrad: https://arxiv.org/abs/1711.06959.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eta (float): initial learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): small constant for numerical stability (default: 1e-5)

    Example:
        >>> optimizer = BPGrad(model.parameters(), eta=1, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.zero_grad()
        >>> loss_value = loss_fn(model(input), target)
        >>> loss_value.backward()
        >>> optimizer.step(lambda: float(loss_value))

    .. note::
        In order to compute the step-size, it requires a closure at every step
        that gives the current value of the objective function.

        Implementation notes: simplification of Algorithm 1 from https://arxiv.org/abs/1711.06959.
        The authors recommend to use N=1 in practice.
        This implies m=1 and therefore rho=0.
        The update on `v` can thus be simplified to `v_{t+1} = mu v_t - (f(x_t) / (L * ||g_t||_2)) * g_t`,
        where g_t is the (stochastic) gradient of f at x=x_t.

        For more details, see:
        https://arxiv.org/abs/1711.06959.
    """

    def __init__(self, params, eta=required, momentum=0, weight_decay=0, eps=1e-5):
        if eta is not required and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(eta=eta, momentum=momentum, weight_decay=weight_decay)
        super(BPGrad, self).__init__(params, defaults)
        self.eps = eps

        for group in self.param_groups:
            group['L'] = 1. / group['eta']
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['v'] = torch.zeros_like(p.data, requires_grad=False)

    torch.autograd.no_grad()
    def step(self, closure):

        obj = float(closure())

        for group in self.param_groups:
            wd = group['weight_decay']
            if wd:
                for p in group['params']:
                    obj += 0.5 * wd * p.data.norm() ** 2
                    p.grad.data += wd * p.data

        grad_sqrd_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                grad_sqrd_norm += p.grad.data.norm() ** 2

        step_size = float(obj / (torch.sqrt(grad_sqrd_norm) + self.eps))

        for group in self.param_groups:
            L = group['L']
            mu = group['momentum']
            for p in group['params']:
                v = self.state[p]['v']
                v *= mu
                v -= step_size / L * p.grad.data
                p.data += v

        self.gamma = step_size
