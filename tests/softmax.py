import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
import math

def softmax(Z):
    Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return Z / Z.sum(axis=-1, keepdims=True)

K = np.random.randn(50, 100, 192//3)
Q = np.random.randn(50, 192//3, 100)

k = torch.tensor(K, requires_grad=True)
q = torch.tensor(Q, requires_grad=True)

K_ = ndl.Tensor(K, device=ndl.cpu(), requires_grad=True)
Q_ = ndl.Tensor(Q, device=ndl.cpu(), requires_grad=True)

res = softmax(K @ Q / math.sqrt(192//3 // 3))
res_ndl = ndl.nn.Softmax()(ndl.ops.batch_matmul(K_, Q_) / (192//3 // 3)**0.5)
res_torch = nn.Softmax(dim=-1)(torch.bmm(k, q) / math.sqrt(192//3 // 3))

print("res_ndl", np.linalg.norm(res-res_ndl.numpy()))
print("res_torch", np.linalg.norm(res_torch.detach().numpy()-res_ndl.numpy()))

res_torch.backward(torch.ones_like(res_torch))
res_ndl.backward()

print("K.grad", np.linalg.norm(k.grad.detach().numpy()-K_.grad.numpy()))