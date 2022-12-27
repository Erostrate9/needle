import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l

class Dense(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, weight, bias,
                 **kwargs):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, X):
        return X@self.weight + self.bias

class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.dense1 = Dense(w1, b1)
        self.relu = nn.ReLU()
        self.dense2 = Dense(w2, b2)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):  #@save
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


ffn_ = ndl.nn.PositionWiseFFN(4, 4, 8, device=ndl.cpu())
one = ndl.init.ones(2, 3, 4 , device=ndl.cpu(), dtype="float32")
y_ = ffn_(one)
print(y_.shape)
w1 = torch.tensor(ffn_.dense1.weight.numpy())
b1 = torch.tensor(ffn_.dense1.bias.numpy())
w2 = torch.tensor(ffn_.dense2.weight.numpy())
b2 = torch.tensor(ffn_.dense2.bias.numpy())
ffn = PositionWiseFFN(w1, b1, w2, b2)
ffn.eval()
y = ffn(torch.ones((2, 3, 4)))
print(y.shape)
print("PositionWiseFFN", np.linalg.norm(y_.numpy()-y.detach().numpy()))

### add_norm
add_norm_ = ndl.nn.AddNorm()
y_ = add_norm_(one, one)
add_norm = AddNorm(4, 0.5)
y = add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4)))
print(y.shape)
###

### TransformerEncoderBlock
device = ndl.cpu()
B, T, d = 2, 100, 24
X = torch.randn(2, 100, 24,dtype=torch.float32)
valid_lens = torch.tensor([3, 2])
encoder_blk = d2l.TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.eval()
y = encoder_blk(X, valid_lens)
print(y.shape)

num_heads = 8
attn = nn.MultiheadAttention(24, num_heads, bias = False, batch_first= True)
M = torch.triu(-float("inf") * torch.ones(T,T), 1)
mask = ndl.Tensor(M.numpy(), device=device)
W_KQV = ndl.Tensor(attn.in_proj_weight.detach().numpy().T, device=device)
W_out = ndl.Tensor(attn.out_proj.weight.detach().numpy().T, device=device)
ffn_num_input = 24
ffn_num_hiddens = 48
num_hiddens = 24
eps = 1e-5
dropout = 0.5
encoder_blk_ = ndl.nn.TransformerEncoderBlock_test(mask, num_heads, W_KQV, W_out,
                ffn_num_input, ffn_num_hiddens, num_hiddens,
                 eps, dropout, device=device, dtype="float32")

X_ = ndl.Tensor(X.numpy(), device=device, dtype="float32")
y_ = encoder_blk_(X_)
print(y_.shape)

trans = nn.TransformerEncoderLayer(d, num_heads, dim_feedforward=48,
                                   dropout=0.5, batch_first=True)
trans.linear1.bias.data.zero_()
trans.linear2.bias.data.zero_()
y_torch = trans(X)
print(y_torch.shape)
print(np.linalg.norm(y_.numpy()))
print(np.linalg.norm(y_torch.detach().numpy()))
###

### TransformerEncoder
valid_lens = torch.tensor([3, 2])
encoder = d2l.TransformerEncoder(200, 24, 48, 8, 2, 0.5)
y = encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)
print(y.shape)
###
