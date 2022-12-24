"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if (param.grad is None) or (not param.requires_grad):
                continue

            new_u = self.u.get(param, 0) * self.momentum + \
                    (1 - self.momentum) * \
                    (param.grad.data + self.weight_decay * param.data)

            param.data = ndl.Tensor((param.data - self.lr * new_u),
                                    device=param.data.device, dtype=param.data.dtype, requires_grad=False).data

            self.u[param] = new_u
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
          # if (param.requires_grad is None) or (param.requires_grad==False):
          #   continue
          wd_grad = self.weight_decay * param.data + param.grad.data
          new_u = (self.beta1 * self.m.get(param, 0) + \
            (1-self.beta1) * wd_grad).data
          new_v = (self.beta2 * self.v.get(param, 0) + \
            (1-self.beta2) * (wd_grad**2)).data
          self.m[param] = new_u
          self.v[param] = new_v
          # bias correction
          new_u /= (1-self.beta1 ** self.t)
          new_v /= (1-self.beta2 ** self.t)
          delta = self.lr * new_u / (new_v**0.5 + self.eps)
          param.data = ndl.Tensor((param.data - delta),
            device=param.data.device, dtype=param.data.dtype).data
        ### END YOUR SOLUTION
