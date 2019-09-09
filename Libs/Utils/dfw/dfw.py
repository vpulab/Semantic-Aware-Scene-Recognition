import torch
import torch.optim as optim

from torch.optim.optimizer import required
from collections import defaultdict


class DFW(optim.Optimizer):
    r"""
    Implements Deep Frank Wolfe: https://openreview.net/forum?id=SyVU6s05K7.
    Nesterov momentum is the *standard formula*, and differs
    from pytorch NAG implementation.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eta (float): initial learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): small constant for numerical stability (default: 1e-5)

    Example:
        >>> optimizer = DFW(model.parameters(), eta=1, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.zero_grad()
        >>> loss_value = loss_fn(model(input), target)
        >>> loss_value.backward()
        >>> optimizer.step(lambda: float(loss_value))

    .. note::
        This optimizer has been designed for convex piecewise linear loss functions only,
        and should be used accordingly.

        In order to compute the step-size, it requires a closure at every step
        that gives the current value of the loss function (without the regularization).

        For more details, see:
        https://openreview.net/forum?id=SyVU6s05K7.
    """

    def __init__(self, params, eta=required, momentum=0, weight_decay=0, eps=1e-5):
        if eta is not required and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(eta=eta, momentum=momentum, weight_decay=weight_decay)
        super(DFW, self).__init__(params, defaults)
        self.eps = eps

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

    torch.autograd.no_grad()
    def step(self, closure):
        loss = float(closure())

        w_dict = defaultdict(dict)
        for group in self.param_groups:
            wd = group['weight_decay']
            for param in group['params']:
                w_dict[param]['delta_t'] = param.grad.data
                w_dict[param]['r_t'] = wd * param.data

        self._line_search(loss, w_dict)

        for group in self.param_groups:
            eta = group['eta']
            mu = group['momentum']
            for param in group['params']:
                state = self.state[param]
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']

                param.data -= eta * (r_t + self.gamma * delta_t)

                if mu:
                    z_t = state['momentum_buffer']
                    z_t *= mu
                    z_t -= eta * self.gamma * (delta_t + r_t)
                    param.data += mu * z_t


    torch.autograd.no_grad()
    def _line_search(self, loss, w_dict):
        """
        Computes the line search in closed form.
        """

        num = loss
        denom = 0

        for group in self.param_groups:
            eta = group['eta']
            for param in group['params']:
                delta_t, r_t = w_dict[param]['delta_t'], w_dict[param]['r_t']
                num -= eta * torch.sum(delta_t * r_t)
                denom += eta * delta_t.norm() ** 2

        self.gamma = float((num / (denom + self.eps)).clamp(min=0, max=1))
        # print(eta * self.gamma)
