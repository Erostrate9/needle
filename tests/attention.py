import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn

B, T, d = 50, 100, 64
X = torch.randn(B, T, d, dtype=torch.float32)
M = torch.triu(-float("inf") * torch.ones(T,T), 1)

heads = 4
attn = nn.MultiheadAttention(d, heads, bias = False, batch_first= True)

mask = ndl.Tensor(M.numpy(), device=ndl.cpu())
W_KQV = ndl.Tensor(attn.in_proj_weight.detach().numpy().T, device=ndl.cpu())
W_out = ndl.Tensor(attn.out_proj.weight.detach().numpy().T, device=ndl.cpu())
X_ = ndl.Tensor(X.numpy(), device=ndl.cpu())

Y_, A_ = attn(X, X, X, attn_mask = M)
Y, A = ndl.nn.MultiheadAttention(mask, heads, W_KQV, W_out,
                                 device=ndl.cpu())(X_)

print(np.linalg.norm(Y.numpy()-Y_.detach().numpy()))