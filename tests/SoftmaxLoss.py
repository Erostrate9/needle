import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn

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
input = torch.randn((64, 10, 201), requires_grad=False).permute((0,2,1))
target = torch.empty((64, 10), dtype=torch.long).random_(201)
loss.reduction = 'none'
output = loss(input, target)
print("Pytroch target.shape:{0}, output.shape:{1}. {2}".format(target.shape, output.shape, loss.reduction))
loss_ = ndl.nn.SoftmaxLoss()
input_ = ndl.Tensor(input.detach().numpy(), device=device)
target_ = ndl.Tensor(target.detach().numpy(), device=device)
loss_.reduction = 'none'
print("input_.shape", input_.shape)
print("target_.shape", target_.shape)
output_ = loss_(input_, target_)
print("needle target.shape:{0}, output.shape:{1}. {2}".format(target_.shape, output_.shape, loss.reduction))
print("loss:", np.linalg.norm(output_.numpy() - output.detach().numpy()))


# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn((64, 10, 201), requires_grad=False).permute((0,2,1))
target = torch.empty((64, 10), dtype=torch.long).random_(201)
loss.reduction = 'mean'
output = loss(input, target)
print("Pytroch target.shape:{0}, output.shape:{1}. {2}".format(target.shape, output.shape, loss.reduction))
loss_ = ndl.nn.SoftmaxLoss()
input_ = ndl.Tensor(input.detach().numpy(), device=device)
target_ = ndl.Tensor(target.detach().numpy(), device=device)
loss_.reduction = 'mean'
print("input_.shape", input_.shape)
print("target_.shape", target_.shape)
output_ = loss_(input_, target_)
print("needle target.shape:{0}, output.shape:{1}. {2}".format(target_.shape, output_.shape, loss.reduction))
print("loss:", np.linalg.norm(output_.numpy() - output.detach().numpy()))

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn((64, 10, 201), requires_grad=False).permute((0,2,1))
target = torch.empty((64, 10), dtype=torch.long).random_(201)
loss.reduction = 'sum'
output = loss(input, target)
print("Pytroch target.shape:{0}, output.shape:{1}. {2}".format(target.shape, output.shape, loss.reduction))
loss_ = ndl.nn.SoftmaxLoss()
input_ = ndl.Tensor(input.detach().numpy(), device=device)
target_ = ndl.Tensor(target.detach().numpy(), device=device)
loss_.reduction = 'sum'
print("input_.shape", input_.shape)
print("target_.shape", target_.shape)
output_ = loss_(input_, target_)
print("needle target.shape:{0}, output.shape:{1}. {2}".format(target_.shape, output_.shape, loss.reduction))
print("loss:", np.linalg.norm(output_.numpy() - output.detach().numpy()))

