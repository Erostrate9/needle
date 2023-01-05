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

y = X[0]
y_ = X_[0,:,:]
y = X.sum()
y_ = X_.sum()
print(type(y))
print(type(y_))
print(np.linalg.norm(y_.numpy() - y.detach().numpy()))

