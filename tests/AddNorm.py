import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

device = ndl.cpu()
one = ndl.init.ones(2, 3, 4 , device=device, dtype="float32")
X = torch.normal(0,1, (2,3,4), dtype=torch.float32)
Y = torch.normal(0,1, (2,3,4), dtype=torch.float32)
X_ = ndl.Tensor(X.detach().numpy(), device=device, dtype="float32")
Y_ = ndl.Tensor(Y.detach().numpy(), device=device, dtype="float32")

### add_norm
norm_shape = 4
dropout=0
add_norm = AddNorm(norm_shape, dropout=dropout)
y = add_norm(X, Y)

add_norm_ = ndl.nn.AddNorm(dropout=dropout)
y_ = add_norm_(X_, Y_)

print(np.linalg.norm(y_.numpy()-y.detach().numpy()))
###