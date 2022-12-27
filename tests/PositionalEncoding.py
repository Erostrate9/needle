import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
from d2l import torch as d2l
import torch
import torch.nn as nn
import numpy as np

encoding_dim, num_steps = 32, 60
pos_encoding = d2l.PositionalEncoding(encoding_dim, 0)

X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
# d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
#          figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])

pos_encoding_ = ndl.nn.PositionalEncoding(encoding_dim, 0)
print("pos_encoding.P:", np.linalg.norm(pos_encoding.P.detach().numpy()-pos_encoding_.P))

P_ = pos_encoding_.P[:, :X.shape[1], :]
print("P:", np.linalg.norm(P.detach().numpy()-P_))