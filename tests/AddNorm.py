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
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, Z):
        return (Z - Z.mean(axis=-1, keepdims=True)) / torch.sqrt(Z.var(axis=-1, keepdims=True, unbiased=True) + self.eps)
class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNorm(eps=1e-5)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

device = ndl.cpu()

X = torch.randn((2, 100, 24), dtype=torch.float32)
Y = torch.randn((2, 100, 24), dtype=torch.float32)
X_ = ndl.Tensor(X.detach().numpy(), device=device, dtype="float32")
Y_ = ndl.Tensor(Y.detach().numpy(), device=device, dtype="float32")

### add_norm
norm_shape = [100, 24]
dropout = 0
add_norm = AddNorm(norm_shape, dropout=dropout)
y = add_norm(X, Y)

add_norm_ = ndl.nn.AddNorm(dropout=dropout)
y_ = add_norm_(X_, Y_)
print(y.shape)
print(y_.shape)
print(np.linalg.norm(y_.numpy()-y.detach().numpy()))

###