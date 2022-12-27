import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
from d2l import torch as d2l
import torch
import torch.nn as nn
import numpy as np

batch_size = 2
num_queries = 4
query_size = 100
num_hiddens = 100
bias = False

X = torch.ones((batch_size, num_queries, num_hiddens),dtype=torch.float32)
X_ = ndl.Tensor(X.detach().numpy(), device=ndl.cpu(), dtype="float32")

W_q = nn.Linear(query_size, num_hiddens, bias=bias)
W_q_ = ndl.nn.Linear(query_size, num_hiddens, bias=bias, device=ndl.cpu(), dtype="float32")
W_q.weight = torch.nn.Parameter(torch.tensor(W_q_.weight.numpy(), dtype=torch.float32))


y = W_q(X)
y_ = W_q_(X_)
print(np.linalg.norm(y.detach().numpy() - y_.numpy()))

y = X @ W_q.weight.T
y_ = X_ @ W_q_.weight
print(np.linalg.norm(y.detach().numpy() - y_.numpy()))