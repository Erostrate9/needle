import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np

def softmax(Z):
    Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return Z / Z.sum(axis=-1, keepdims=True)

K = np.random.randn(50, 100, 192//3)
Q = np.random.randn(50, 192//3, 100)

K_ = ndl.Tensor(K, device=ndl.cpu(), requires_grad=False)
Q_ = ndl.Tensor(Q, device=ndl.cpu(), requires_grad=False)

res = softmax(K @ Q / np.sqrt(192//3 // 3))
res_ndl = ndl.ops.softmax(ndl.ops.batch_matmul(K_, Q_) / (192//3 // 3)**0.5)
print("res_ndl", res_ndl.shape)
print(np.linalg.norm(res-res_ndl.numpy()))