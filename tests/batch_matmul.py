import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np

C = np.random.randn(5,4,4,4,10,3)
D = np.random.randn(5,4,4,4,3,6)

C_ = ndl.Tensor(C, device=ndl.cpu(), requires_grad=False)
D_ = ndl.Tensor(D, device=ndl.cpu(), requires_grad=False)

res = C@D
res_ndl = ndl.ops.batch_matmul(C_, D_)

print(np.linalg.norm(res-res_ndl.numpy()))