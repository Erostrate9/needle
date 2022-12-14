import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l

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

class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class LayerNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, Z):
        return (Z - Z.mean(axis=-1, keepdims=True)) / torch.sqrt(Z.var(axis=-1, keepdims=True, unbiased=True) + self.eps)

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNorm(eps=1e-5)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            bias=use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        ###
        self.X = None
        self.attn = None
        self.Y = None
        self.ffnY = None
        self.res = None

    def forward(self, X, valid_lens):
        self.X = X.detach().numpy()
        attn = self.attention(X, X, X, valid_lens)
        self.attn = attn.detach().numpy()
        Y = self.addnorm1(X, attn)
        self.Y = Y.detach().numpy()
        ffnY = self.ffn(Y)
        self.ffnY = ffnY.detach().numpy()
        res = self.addnorm2(Y, ffnY)
        self.res = res.detach().numpy()
        return res

use_bias = False
dropout = 0
X = torch.randn((2, 100, 24)).type(torch.float32)
valid_lens = torch.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, dropout, use_bias=use_bias)
# print(y.shape)

device = ndl.cpu()
X_ = ndl.Tensor(X.detach().numpy(), device=device, dtype="float32")
valid_lens_ = valid_lens.detach().numpy()
encoder_blk_ = ndl.nn.TransformerEncoderBlock(24, 24, 24, 24, 24, 48, 8, dropout, use_bias=use_bias, device=device, dtype="float32")
encoder_blk_.eval()

# multiHeadAttention_weight
encoder_blk.attention.W_q.weight = torch.nn.Parameter(torch.tensor(encoder_blk_.attention.W_q.weight.numpy().T, dtype=torch.float32))
encoder_blk.attention.W_k.weight = torch.nn.Parameter(torch.tensor(encoder_blk_.attention.W_k.weight.numpy().T, dtype=torch.float32))
encoder_blk.attention.W_v.weight = torch.nn.Parameter(torch.tensor(encoder_blk_.attention.W_v.weight.numpy().T, dtype=torch.float32))
encoder_blk.attention.W_o.weight = torch.nn.Parameter(torch.tensor(encoder_blk_.attention.W_o.weight.numpy().T, dtype=torch.float32))
if use_bias:
    encoder_blk.attention.W_q.bias = torch.nn.Parameter(
        torch.tensor(encoder_blk_.attention.W_q.bias.numpy(), dtype=torch.float32))
    encoder_blk.attention.W_k.bias = torch.nn.Parameter(
        torch.tensor(encoder_blk_.attention.W_k.bias.numpy(), dtype=torch.float32))
    encoder_blk.attention.W_v.bias = torch.nn.Parameter(
        torch.tensor(encoder_blk_.attention.W_v.bias.numpy(), dtype=torch.float32))
    encoder_blk.attention.W_o.bias = torch.nn.Parameter(
        torch.tensor(encoder_blk_.attention.W_o.bias.numpy(), dtype=torch.float32))
# ffn_weight
encoder_blk.ffn.dense1.weight = torch.nn.Parameter(torch.tensor(encoder_blk_.ffn.dense1.weight.numpy().T, dtype=torch.float32))
encoder_blk.ffn.dense2.weight = torch.nn.Parameter(torch.tensor(encoder_blk_.ffn.dense2.weight.numpy().T, dtype=torch.float32))
encoder_blk.ffn.dense1.bias = torch.nn.Parameter(torch.tensor(encoder_blk_.ffn.dense1.bias.numpy(), dtype=torch.float32))
encoder_blk.ffn.dense2.bias = torch.nn.Parameter(torch.tensor(encoder_blk_.ffn.dense2.bias.numpy(), dtype=torch.float32))
encoder_blk.eval()

n = 5
for i in range(n):
    X = encoder_blk(X, valid_lens)
    X_ = encoder_blk_(X_, valid_lens_)
    # print(y_.shape)
    # print("EncoderBlock.X", np.linalg.norm(encoder_blk.X - encoder_blk_.X))
    # print("EncoderBlock.attn", np.linalg.norm(encoder_blk.attn - encoder_blk_.attn))
    # print("EncoderBlock.Y", np.linalg.norm(encoder_blk.Y - encoder_blk_.Y))
    # print("EncoderBlock.ffnY", np.linalg.norm(encoder_blk.ffnY - encoder_blk_.ffnY))
    # print("EncoderBlock.res", np.linalg.norm(encoder_blk.res - encoder_blk_.res))
    print("EncoderBlock", np.linalg.norm(X_.numpy()-X.detach().numpy()))