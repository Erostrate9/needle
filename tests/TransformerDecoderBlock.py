import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l
import math

class MultiHeadAttention(nn.Module):
    """MultiHeadAttention"""
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
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNorm(eps=1e-5)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout=0, use_bias=False, **kwargs):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            bias=use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)
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


class TransformerEncoder(d2l.Encoder):
    """Transformer Encoder"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers=1, dropout=0, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                TransformerEncoderBlock(key_size, query_size, value_size, num_hiddens,
                            ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

        ### debug
        self.X = None
        self.ebd = None
        self.X = []
        self.y = []

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        ebd = self.embedding(X)
        self.ebd = ebd.detach().numpy()
        X = self.pos_encoding(ebd * math.sqrt(self.num_hiddens))
        self.X.append(X)
        ###
        ###
        self.attention_weights = [None] * len(self.blks)
        X = X.type(torch.float32)
        for i, blk in enumerate(self.blks):
            # self.X.append(X.detach().numpy())
            X = blk(X, valid_lens)
            self.y.append(X.detach().numpy())
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
            enc_outputs, enc_valid_lens = state[0], state[1]
            # During training, all the tokens of any output sequence are processed
            # at the same time, so state[2][self.i] is None as initialized. When
            # decoding any output sequence token by token during prediction,
            # state[2][self.i] contains representations of the decoded output at
            # the i-th block up to the current time step
            if state[2][self.i] is None:
                key_values = X
            else:
                key_values = torch.cat((state[2][self.i], X), dim=1)
            state[2][self.i] = key_values
            if self.training:
                batch_size, num_steps, _ = X.shape
                # Shape of dec_valid_lens: (batch_size, num_steps), where every
                # row is [1, 2, ..., num_steps]
                dec_valid_lens = torch.arange(
                    1, num_steps + 1, device=X.device).repeat(batch_size, 1)
            else:
                dec_valid_lens = None
            # Self-attention
            X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
            Y = self.addnorm1(X, X2)
            # Encoder-decoder attention. Shape of enc_outputs:
            # (batch_size, num_steps, num_hiddens)
            Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
            Z = self.addnorm2(Y, Y2)
            return self.addnorm3(Z, self.ffn(Z)), state



dropout = 0
use_bias = True
device = ndl.cpu()

X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])

X_ = ndl.Tensor(X.detach().numpy(), device=device, dtype="float32")
valid_lens_ = valid_lens.detach().numpy()


# encoder_blk_
encoder_blk_ = ndl.nn.TransformerEncoderBlock(24, 24, 24, 24, 24, 48, 8, dropout, use_bias=use_bias, device=device, dtype="float32")
encoder_blk_.eval()

# encoder_blk
encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, 24, 48, 8, dropout, use_bias=use_bias)
###
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
###
encoder_blk.eval()

decoder_blk_ = ndl.nn.TransformerDecoderBlock(24, 24, 24, 24, 24, 48, 8, dropout, 0, use_bias=use_bias, device=device, dtype="float32")
decoder_blk_.eval()

decoder_blk = TransformerDecoderBlock(24, 24, 24, 24, 24, 48, 8, dropout, 0)
###
# multiHeadAttention_weight
decoder_blk.attention1.W_q.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.attention1.W_q.weight.numpy().T, dtype=torch.float32))
decoder_blk.attention1.W_k.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.attention1.W_k.weight.numpy().T, dtype=torch.float32))
decoder_blk.attention1.W_v.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.attention1.W_v.weight.numpy().T, dtype=torch.float32))
decoder_blk.attention1.W_o.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.attention1.W_o.weight.numpy().T, dtype=torch.float32))
decoder_blk.attention2.W_q.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.attention2.W_q.weight.numpy().T, dtype=torch.float32))
decoder_blk.attention2.W_k.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.attention2.W_k.weight.numpy().T, dtype=torch.float32))
decoder_blk.attention2.W_v.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.attention2.W_v.weight.numpy().T, dtype=torch.float32))
decoder_blk.attention2.W_o.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.attention2.W_o.weight.numpy().T, dtype=torch.float32))
if use_bias:
    decoder_blk.attention1.W_q.bias = torch.nn.Parameter(
        torch.tensor(decoder_blk_.attention1.W_q.bias.numpy(), dtype=torch.float32))
    decoder_blk.attention1.W_k.bias = torch.nn.Parameter(
        torch.tensor(decoder_blk_.attention1.W_k.bias.numpy(), dtype=torch.float32))
    decoder_blk.attention1.W_v.bias = torch.nn.Parameter(
        torch.tensor(decoder_blk_.attention1.W_v.bias.numpy(), dtype=torch.float32))
    decoder_blk.attention1.W_o.bias = torch.nn.Parameter(
        torch.tensor(decoder_blk_.attention1.W_o.bias.numpy(), dtype=torch.float32))
    decoder_blk.attention2.W_q.bias = torch.nn.Parameter(
        torch.tensor(decoder_blk_.attention2.W_q.bias.numpy(), dtype=torch.float32))
    decoder_blk.attention2.W_k.bias = torch.nn.Parameter(
        torch.tensor(decoder_blk_.attention2.W_k.bias.numpy(), dtype=torch.float32))
    decoder_blk.attention2.W_v.bias = torch.nn.Parameter(
        torch.tensor(decoder_blk_.attention2.W_v.bias.numpy(), dtype=torch.float32))
    decoder_blk.attention2.W_o.bias = torch.nn.Parameter(
        torch.tensor(decoder_blk_.attention2.W_o.bias.numpy(), dtype=torch.float32))
# ffn_weight
decoder_blk.ffn.dense1.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.ffn.dense1.weight.numpy().T, dtype=torch.float32))
decoder_blk.ffn.dense2.weight = torch.nn.Parameter(torch.tensor(decoder_blk_.ffn.dense2.weight.numpy().T, dtype=torch.float32))
decoder_blk.ffn.dense1.bias = torch.nn.Parameter(torch.tensor(decoder_blk_.ffn.dense1.bias.numpy(), dtype=torch.float32))
decoder_blk.ffn.dense2.bias = torch.nn.Parameter(torch.tensor(decoder_blk_.ffn.dense2.bias.numpy(), dtype=torch.float32))
###
decoder_blk.eval()



X = torch.randn((2, 100, 24), dtype=torch.float32)
X_ = ndl.Tensor(X, device=device, dtype="float32")

state = [encoder_blk(X, valid_lens), valid_lens, [None]]
state_ = [encoder_blk_(X_, valid_lens_), valid_lens_, [None]]
y, s = decoder_blk(X, state)
y_, s_ = decoder_blk_(X_, state_)
print(y.shape)
print(y_.shape)
print("decoder block:", np.linalg.norm(y.detach().numpy() - y_.numpy()))
