import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
from d2l import torch as d2l
import torch
import numpy as np

queries, keys = torch.normal(0, 1, (2, 10, 20)), torch.ones((2, 10, 20))
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = torch.tensor([2, 6])

queries = torch.normal(0, 1, (2, 10, 20))
attention = d2l.DotProductAttention(dropout=0.5)
attention.eval()

y = attention(queries, keys, values, valid_lens)

queries_ = ndl.Tensor(queries.detach().numpy(), device=ndl.cpu(), dtype="float32")
keys_ = ndl.Tensor(keys.detach().numpy(), device=ndl.cpu(), dtype="float32")
values_ = ndl.Tensor(values.detach().numpy(), device=ndl.cpu(), dtype="float32")
valid_lens_ = valid_lens.detach().numpy()
attention_ = ndl.nn.DotProductAttention(dropout=0.5)
attention_.eval()
y_ = attention_(queries_, keys_, values_, valid_lens_)
print(y)
print(y_)
print(np.linalg.norm(y.detach().numpy()-y_.numpy()))