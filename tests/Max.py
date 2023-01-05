import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
from torch.autograd import Variable

def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad], [g for g in numerical_grad]




device = ndl.cpu()
axes = None

# B = torch.tensor([[1,2,3,3,4,5,5,2]], dtype=torch.float32)
# x = Variable(B, requires_grad=True)
# z, _ = torch.max(x, dim=axes)
# z.backward()
# print(x.grad)

# _A = B.detach().numpy().astype(np.float32)
_A = np.random.randn(3,3,3).astype(np.float32)
A = ndl.Tensor(_A, device=device)

# backward_grad, numerical_grad = backward_check(ndl.summation, A, axes=axes)
backward_grad, numerical_grad = backward_check(ndl.max, A, axes=axes)
# print("pytorch grad:", np.linalg.norm(numerical_grad[0] - x.grad.detach().numpy()))

