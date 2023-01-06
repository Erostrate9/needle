import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l

class LayerNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = [normalized_shape,] if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps

    def forward(self, Z):
        axis = tuple(range(-len(self.normalized_shape), 0))
        print(axis)
        return (Z - Z.mean(axis=axis, keepdims=True)) / torch.sqrt(Z.var(axis=axis, keepdims=True, unbiased=False) + self.eps)


device = ndl.cpu()
X = torch.randn((2, 5, 100, 24), dtype=torch.float32)
X_ = ndl.Tensor(X.detach().numpy(), device=device, dtype="float32")

normalized_shape = 24
_ln = nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=False)
ln = LayerNorm(normalized_shape, eps=1e-5)
ln_ = ndl.nn.LayerNorm(normalized_shape, 1e-5, unbiased=False)

_y = _ln(X)
y = ln(X)
y_ = ln_(X_)

print(_y.shape)
print(y.shape)
print(y_.shape)


# axis = len(X_.shape)-1
# d = X_.shape[-1]
# e_ = X_.sum(axes=(axis,)).reshape(X.shape[:-1] + (1,)) / d
# e = X.mean(dim=-1, keepdim=True)
#
# var_ = (((X_ - e_.broadcast_to(X_.shape)) ** 2).sum(axes=(axis,)) / d).reshape(X_.shape[:-1] + (1,))
# var = X.var(dim=-1, keepdim=True, unbiased=False)
# print("diff of mean:", np.linalg.norm(e_.numpy() - e.detach().numpy()))
# print("diff of biased var:", np.linalg.norm(var_.numpy() - var.detach().numpy()))

print("diff of y:", np.linalg.norm(_y.detach().numpy()-y.detach().numpy()))
print("diff of y:", np.linalg.norm(y_.numpy()-y.detach().numpy()))