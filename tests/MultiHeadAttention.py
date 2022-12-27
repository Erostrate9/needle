import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
from d2l import torch as d2l
import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        torch.nn.init.kaiming_uniform_(self.W_q.weight)
        torch.nn.init.kaiming_uniform_(self.W_k.weight)
        torch.nn.init.kaiming_uniform_(self.W_v.weight)
        torch.nn.init.kaiming_uniform_(self.W_o.weight)
        ### test
        self.X1 = None
        self.X2 = None
        self.X3 = None
        self.output = None
        self.vl = None

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        self.vl = valid_lens
        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        self.output = output
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        """为了多注意力头的并行计算而变换形状"""
        # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
        # num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        self.X1 = X.detach().numpy()
        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        self.X2 = X.detach().numpy()

        # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X3 = X.reshape(-1, X.shape[2], X.shape[3])
        self.X3 = X3.detach().numpy()
        return X3


    def transpose_output(self, X, num_heads):
        """逆转transpose_qkv函数的操作"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)



num_hiddens, num_heads = 100, 5

batch_size, num_queries = 2, 4
num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens),dtype=torch.float32)
Y = torch.ones((batch_size, num_kvpairs, num_hiddens),dtype=torch.float32)
# d2l.check_shape(attention(X, Y, Y, valid_lens),
#                 (batch_size, num_queries, num_hiddens))
dropout = 0

attention_ = ndl.nn.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                       num_hiddens, num_heads, dropout, device=ndl.cpu(), dtype="float32")
valid_lens_ = valid_lens.detach().numpy()
X_ = ndl.Tensor(X.detach().numpy(), device=ndl.cpu(), dtype="float32")
Y_ = ndl.Tensor(Y.detach().numpy(), device=ndl.cpu(), dtype="float32")


attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, dropout)

attention.W_q.weight = torch.nn.Parameter(torch.tensor(attention_.W_q.weight.numpy().T, dtype=torch.float32))
attention.W_k.weight = torch.nn.Parameter(torch.tensor(attention_.W_k.weight.numpy().T, dtype=torch.float32))
attention.W_v.weight = torch.nn.Parameter(torch.tensor(attention_.W_v.weight.numpy().T, dtype=torch.float32))
attention.W_o.weight = torch.nn.Parameter(torch.tensor(attention_.W_o.weight.numpy().T, dtype=torch.float32))

print("W_q.weight:", np.linalg.norm(attention.W_q.weight.T.detach().numpy()-attention_.W_q.weight.numpy()))
print("W_k.weight:", np.linalg.norm(attention.W_k.weight.T.detach().numpy()-attention_.W_k.weight.numpy()))
print("W_v.weight:", np.linalg.norm(attention.W_v.weight.T.detach().numpy()-attention_.W_v.weight.numpy()))
print("W_o.weight:", np.linalg.norm(attention.W_o.weight.T.detach().numpy()-attention_.W_o.weight.numpy()))

print("X:", np.linalg.norm(X.detach().numpy()-X_.numpy()))
queries = attention.transpose_qkv(attention.W_q(X), attention.num_heads)
queries_ = attention_.transpose_qkv(attention_.W_q(X_))

zq = attention.W_q(X).detach().numpy()
zq_ = attention_.W_q(X_).numpy()
print("W_q.weight:", np.linalg.norm(attention.W_q.weight.T.detach().numpy() - attention_.W_q.weight.numpy()))

print("W_q(X):", np.linalg.norm(zq - zq_))
X1 = X.reshape((X.shape[0], X.shape[1], attention.num_heads, -1))
X1_ = X_.reshape((X_.shape[0], X_.shape[1], attention_.num_heads, -1))
print("X1-X1_:", np.linalg.norm(X1.detach().numpy() - X1_.numpy()))
# print("X1.shape", attention.X1.shape)
# print("X1_.shape", attention_.X1.shape)
# print("X2.shape", attention.X2.shape)
# print("X2_.shape", attention_.X2.shape)
# print("X3.shape", attention.X3.shape)
# print("X3_.shape", attention_.X3.shape)
# print("X1:", np.linalg.norm(attention.X1-attention_.X1))
# print("X2:", np.linalg.norm(attention.X2-attention_.X2))
# print("X3:", np.linalg.norm(attention.X3-attention_.X3))

keys = attention.transpose_qkv(attention.W_k(Y), attention.num_heads)
keys_ = attention_.transpose_qkv(attention_.W_k(Y_))
# print("X1:", np.linalg.norm(attention.X1-attention_.X1))
# print("X2:", np.linalg.norm(attention.X2-attention_.X2))
# print("X3:", np.linalg.norm(attention.X3-attention_.X3))

values = attention.transpose_qkv(attention.W_v(Y), attention.num_heads)
values_ = attention_.transpose_qkv(attention_.W_v(Y_))
# print("X1:", np.linalg.norm(attention.X1-attention_.X1))
# print("X2:", np.linalg.norm(attention.X2-attention_.X2))
# print("X3:", np.linalg.norm(attention.X3-attention_.X3))


print(np.linalg.norm(X.detach().numpy()-X_.numpy()))
print(np.linalg.norm(Y.detach().numpy()-Y_.numpy()))

print(np.linalg.norm(queries.detach().numpy()-queries_.numpy()))
print(np.linalg.norm(keys.detach().numpy()-keys_.numpy()))
print(np.linalg.norm(values.detach().numpy()-values_.numpy()))

attention.eval()
y = attention(X, Y, Y, valid_lens)
print("attn_output.shape:", y.shape)

y_ = attention_(X_, Y_, Y_, valid_lens_)
print("attn_output_.shape:", y_.shape)

print("valid_lens:", np.linalg.norm(attention.vl.detach().numpy()-attention_.vl))
print("output:", np.linalg.norm(attention.output.detach().numpy()-attention_.output.numpy()))
print("attn_output:", np.linalg.norm(y.detach().numpy()-y_.numpy()))