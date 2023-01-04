import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l
import math

device = ndl.cpu()
X = torch.randn((2, 100, 24), dtype=torch.float32)
X_ = ndl.Tensor(X, device=device, dtype="float32")

# y = X[0]
# y_ = X_[0,/:,:]
# y = X.sum()
# y_ = X_.sum()
# print(type(y))
# print(type(y_))
# print(np.linalg.norm(y_.numpy() - y.detach().numpy()))

class SoftmaxLoss(ndl.nn.Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def forward(self, logits: ndl.Tensor, y: ndl.Tensor):
        ### BEGIN YOUR SOLUTION
        # n: batch_num; k: class_num
        n, k, *_ = logits.shape
        y_one_hot = ndl.init.one_hot(k, y, device=y.device, dtype=y.dtype)
        # axes=(1,)
        logsumexp = ndl.ops.logsumexp(logits, axes=(1,))
        z_y = (logits * y_one_hot).sum(axes=(1,))
        if self.reduction == 'none':
            return (logsumexp - z_y)
        elif self.reduction == 'sum':
            return (logsumexp - z_y).sum()
        else:
            # mean
            return (logsumexp - z_y).sum() / n
        ### END YOUR SOLUTION


device = ndl.cpu()
# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
loss.reduction = 'none'
output = loss(input, target)
print("Pytroch target.shape:{0}, output.shape:{1}. {2}".format(target.shape, output.shape, loss.reduction))
loss_ = SoftmaxLoss()
input_ = ndl.Tensor(input.detach().numpy(), device=device)
target_ = ndl.Tensor(target.detach().numpy(), device=device)
loss_.reduction = 'none'
output_ = loss_(input_, target_)
print("needle target.shape:{0}, output.shape:{1}. {2}".format(target_.shape, output_.shape, loss.reduction))
print(np.linalg.norm(output_.numpy() - output.detach().numpy()))

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
loss.reduction = 'mean'
output = loss(input, target)
print("Pytroch target.shape:{0}, output.shape:{1}. {2}".format(target.shape, output.shape, loss.reduction))
loss_ = SoftmaxLoss()
input_ = ndl.Tensor(input.detach().numpy(), device=device)
target_ = ndl.Tensor(target.detach().numpy(), device=device)
loss_.reduction = 'mean'
output_ = loss_(input_, target_)
print("needle target.shape:{0}, output.shape:{1}. {2}".format(target_.shape, output_.shape, loss.reduction))
print(np.linalg.norm(output_.numpy() - output.detach().numpy()))

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
loss.reduction = 'sum'
output = loss(input, target)
print("Pytroch target.shape:{0}, output.shape:{1}. {2}".format(target.shape, output.shape, loss.reduction))
loss_ = SoftmaxLoss()
input_ = ndl.Tensor(input.detach().numpy(), device=device)
target_ = ndl.Tensor(target.detach().numpy(), device=device)
loss_.reduction = 'sum'
output_ = loss_(input_, target_)
print("needle target.shape:{0}, output.shape:{1}. {2}".format(target_.shape, output_.shape, loss.reduction))
print(np.linalg.norm(output_.numpy() - output.detach().numpy()))