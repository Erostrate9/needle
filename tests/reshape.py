import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
from d2l import torch as d2l
import torch
import torch.nn as nn
import numpy as np

x = torch.normal(0, 1, (2, 4, 100))
x_ = ndl.Tensor(x.detach().numpy(), device=ndl.cpu(), dtype="float32")

print(np.linalg.norm(x.reshape(x.shape[0],x.shape[1],5,-1).detach().numpy() - x_.reshape((x_.shape[0],x_.shape[1],5,-1)).numpy()))
