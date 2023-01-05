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
        # self.X1 = None
        # self.X2 = None
        # self.X3 = None
        # self.output = None
        # self.vl = None

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
        # debug
        # self.X1 = queries
        # self.X2 = keys
        # self.X3 = values
        #
        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        # self.vl = valid_lens
        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # self.output = output
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        """为了多注意力头的并行计算而变换形状"""
        # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
        # num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        # self.X1 = X.detach().numpy()
        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        # self.X2 = X.detach().numpy()

        # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X3 = X.reshape(-1, X.shape[2], X.shape[3])
        # self.X3 = X3.detach().numpy()
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
        # self.X = None
        # self.attn = None
        # self.Y = None
        # self.ffnY = None
        # self.res = None

    def forward(self, X, valid_lens):
        # self.X = X.detach().numpy()
        attn = self.attention(X, X, X, valid_lens)
        # self.attn = attn.detach().numpy()
        Y = self.addnorm1(X, attn)
        # self.Y = Y.detach().numpy()
        ffnY = self.ffn(Y)
        # self.ffnY = ffnY.detach().numpy()
        res = self.addnorm2(Y, ffnY)
        # self.res = res.detach().numpy()
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

        # ### debug
        # self.X = None
        # self.ebd = None
        # self.X = []
        # self.y = []

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        ebd = self.embedding(X)
        X = self.pos_encoding(ebd * math.sqrt(self.num_hiddens))
        ###
        ###
        self.attention_weights = [None] * len(self.blks)
        X = X.type(torch.float32)
        for i, blk in enumerate(self.blks):
            # self.X.append(X.detach().numpy())
            X = blk(X, valid_lens)
            # self.y.append(X.detach().numpy())
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, use_bias=True, **kwargs):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(dropout)
        # debug
        self.X = None
        self.enc_outputs = None
        self.enc_valid_lens = None
        self.X2 = None
        self.Y = None
        self.Y2 = None
        self.Z = None
        self.key_values = None
        self.dec_valid_lens = None

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
            # debug
            self.X = X
            self.enc_outputs = enc_outputs
            self.enc_valid_lens = enc_valid_lens
            self.key_values = key_values
            self.dec_valid_lens = dec_valid_lens
            self.X2 = X2
            self.Y = Y
            self.Y2 = Y2
            self.Z = Z
            return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=True, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                TransformerDecoderBlock(key_size, query_size, value_size, num_hiddens,
                              ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i, use_bias=use_bias))
        self.dense = nn.Linear(num_hiddens, vocab_size)
        # debug
        self.X = []
        self.state = []

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]

        ##
        self.X.append(X)
        ##
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            ##
            self.X.append(X)
            self.state.append(state)
            ##
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

def set_encoder_blk(encoder_blk, encoder_blk_, use_bias=True):
    # multiHeadAttention_weight
    encoder_blk.attention.W_q.weight = torch.nn.Parameter(
        torch.tensor(encoder_blk_.attention.W_q.weight.numpy().T, dtype=torch.float32))
    encoder_blk.attention.W_k.weight = torch.nn.Parameter(
        torch.tensor(encoder_blk_.attention.W_k.weight.numpy().T, dtype=torch.float32))
    encoder_blk.attention.W_v.weight = torch.nn.Parameter(
        torch.tensor(encoder_blk_.attention.W_v.weight.numpy().T, dtype=torch.float32))
    encoder_blk.attention.W_o.weight = torch.nn.Parameter(
        torch.tensor(encoder_blk_.attention.W_o.weight.numpy().T, dtype=torch.float32))
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
    encoder_blk.ffn.dense1.weight = torch.nn.Parameter(
        torch.tensor(encoder_blk_.ffn.dense1.weight.numpy().T, dtype=torch.float32))
    encoder_blk.ffn.dense2.weight = torch.nn.Parameter(
        torch.tensor(encoder_blk_.ffn.dense2.weight.numpy().T, dtype=torch.float32))
    encoder_blk.ffn.dense1.bias = torch.nn.Parameter(
        torch.tensor(encoder_blk_.ffn.dense1.bias.numpy(), dtype=torch.float32))
    encoder_blk.ffn.dense2.bias = torch.nn.Parameter(
        torch.tensor(encoder_blk_.ffn.dense2.bias.numpy(), dtype=torch.float32))

def set_encoder(encoder, encoder_, use_bias=True):
    # encoder: pytorch TransformerEncoder
    # encoder_: needle TransformerEncoder
    for i in range(len(encoder.blks)):
        encoder_blk = encoder.blks[i]
        encoder_blk_ = encoder_.blks.modules[i]
        # multiHeadAttention_weight
        set_encoder_blk(encoder_blk, encoder_blk_, use_bias=use_bias)
    encoder.embedding.weight = torch.nn.Parameter(torch.tensor(encoder_.embedding.weight.numpy(), dtype=torch.float32))

def set_decoder(decoder, decoder_, use_bias=True):
    # decoder: pytorch TransformerDecoder
    # decoder_: needle TransformerDecoder
    for i in range(len(decoder.blks)):
        decoder_blk = decoder.blks[i]
        decoder_blk_ = decoder_.blks.modules[i]
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
    decoder.embedding.weight = torch.nn.Parameter(torch.tensor(decoder_.embedding.weight.numpy(), dtype=torch.float32))
    decoder.dense.weight = torch.nn.Parameter(torch.tensor(decoder_.dense.weight.numpy().T, dtype=torch.float32))
    decoder.dense.bias = torch.nn.Parameter(torch.tensor(decoder_.dense.bias.numpy(), dtype=torch.float32))

def to_ndl(tensor, device=None, dtype="float32"):
    return ndl.Tensor(tensor.to(torch.device('cpu')).detach().numpy(), device=device, dtype=dtype)

def check_weight(decoder_blk, decoder_blk_):
    # multiHeadAttention_weight
    print("----------attention1----------")
    print("W_q: ", np.linalg.norm(
        decoder_blk.attention1.W_q.weight.detach().numpy().T - decoder_blk_.attention1.W_q.weight.numpy()))
    print("W_k: ", np.linalg.norm(
        decoder_blk.attention1.W_k.weight.detach().numpy().T - decoder_blk_.attention1.W_k.weight.numpy()))
    print("W_v: ", np.linalg.norm(
        decoder_blk.attention1.W_v.weight.detach().numpy().T - decoder_blk_.attention1.W_v.weight.numpy()))
    print("W_o: ", np.linalg.norm(
        decoder_blk.attention1.W_o.weight.detach().numpy().T - decoder_blk_.attention1.W_o.weight.numpy()))
    if use_bias:
        print("W_q.bias: ", np.linalg.norm(
            decoder_blk.attention1.W_q.bias.detach().numpy() - decoder_blk_.attention1.W_q.bias.numpy()))
        print("W_k.bias: ", np.linalg.norm(
            decoder_blk.attention1.W_k.bias.detach().numpy() - decoder_blk_.attention1.W_k.bias.numpy()))
        print("W_v.bias: ", np.linalg.norm(
            decoder_blk.attention1.W_v.bias.detach().numpy() - decoder_blk_.attention1.W_v.bias.numpy()))
        print("W_o.bias: ", np.linalg.norm(
            decoder_blk.attention1.W_o.bias.detach().numpy() - decoder_blk_.attention1.W_o.bias.numpy()))
    print("----------attention2----------")
    print("W_q: ", np.linalg.norm(
        decoder_blk.attention2.W_q.weight.detach().numpy().T - decoder_blk_.attention2.W_q.weight.numpy()))
    print("W_k: ", np.linalg.norm(
        decoder_blk.attention2.W_k.weight.detach().numpy().T - decoder_blk_.attention2.W_k.weight.numpy()))
    print("W_v: ", np.linalg.norm(
        decoder_blk.attention2.W_v.weight.detach().numpy().T - decoder_blk_.attention2.W_v.weight.numpy()))
    print("W_o: ", np.linalg.norm(
        decoder_blk.attention2.W_o.weight.detach().numpy().T - decoder_blk_.attention2.W_o.weight.numpy()))
    if use_bias:
        print("W_q.bias: ", np.linalg.norm(
            decoder_blk.attention2.W_q.bias.detach().numpy() - decoder_blk_.attention2.W_q.bias.numpy()))
        print("W_k.bias: ", np.linalg.norm(
            decoder_blk.attention2.W_k.bias.detach().numpy() - decoder_blk_.attention2.W_k.bias.numpy()))
        print("W_v.bias: ", np.linalg.norm(
            decoder_blk.attention2.W_v.bias.detach().numpy() - decoder_blk_.attention2.W_v.bias.numpy()))
        print("W_o.bias: ", np.linalg.norm(
            decoder_blk.attention2.W_o.bias.detach().numpy() - decoder_blk_.attention2.W_o.bias.numpy()))
    # ffn_weight
    print("dense1.weight: ", np.linalg.norm(
        decoder_blk.ffn.dense1.weight.detach().numpy().T - decoder_blk_.ffn.dense1.weight.detach().numpy()))
    print("dense2.weight: ", np.linalg.norm(
        decoder_blk.ffn.dense2.weight.detach().numpy().T - decoder_blk_.ffn.dense2.weight.detach().numpy()))
    print("dense1.bias: ", np.linalg.norm(
        decoder_blk.ffn.dense1.bias.detach().numpy() - decoder_blk_.ffn.dense1.bias.numpy()))
    print("dense2.bias: ", np.linalg.norm(
        decoder_blk.ffn.dense2.bias.detach().numpy() - decoder_blk_.ffn.dense2.bias.numpy()))


use_bias = True
device_ = ndl.cuda()
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0, 64, 10
lr, num_epochs, device = 0.005, 200, torch.device('cuda')
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32


train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout, use_bias=use_bias)
decoder_ = ndl.nn.TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout,
    use_bias=use_bias, device=device_, dtype="float32")



encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout, use_bias=use_bias)
encoder_ = ndl.nn.TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout,
    use_bias=use_bias, device=device_, dtype="float32")

set_decoder(decoder, decoder_, use_bias=use_bias)
set_encoder(encoder, encoder_, use_bias=use_bias)

encoder.eval()
encoder_.eval()
decoder.eval()
decoder_.eval()

encoder.to(device)
decoder.to(device)

net = EncoderDecoder(encoder, decoder)
net_ = ndl.nn.EncoderDecoder(encoder_, decoder_)

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
loss = d2l.MaskedSoftmaxCELoss()
loss_ = ndl.nn.MaskedSoftmaxCELoss()

for idx, batch in enumerate(train_iter):
    X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
    bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                       device=device).reshape(-1, 1)
    dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
    Y_hat, _ = net(X, dec_input, X_valid_len)
    print(Y_hat.shape)

    X_, X_valid_len_, Y_, Y_valid_len_ = to_ndl(X, device_), to_ndl(X_valid_len).numpy(), to_ndl(Y, device_), to_ndl(Y_valid_len).numpy()
    bos_ = to_ndl(bos, device=device_)
    dec_input_ = to_ndl(dec_input, device=device_)
    Y_hat_, __ = net_(X_, dec_input_, X_valid_len_)
    print(Y_hat_.shape)
    # print('diff of enc_outputs: ', np.linalg.norm(net.enc_outputs.detach().numpy() - net_.enc_outputs.numpy()))
    # print('diff of dec_state[0]: ', np.linalg.norm(net.dec_state[0].detach().numpy() - net_.dec_state[0].numpy()))
    print('batch{0}, diff of Y_hat: {1}'.format(idx, np.linalg.norm(Y_hat.detach().to(torch.device('cpu')).numpy() - Y_hat_.numpy())) )
    l = loss(Y_hat, Y, Y_valid_len)
    l_ = loss_(Y_hat_, Y_, Y_valid_len_)
    print('batch{0}, diff of loss: {1}'.format(idx, np.linalg.norm(
        l.detach().to(torch.device('cpu')).numpy() - l_.numpy())))
    # for i in range(len(net.decoder.X)):
    #     print('diff of X_{0}: {1}'.format(i, np.linalg.norm(net.decoder.X[i].detach().numpy() - net_.decoder.X[i].numpy())) )
    # break




#
# for i in range(len(decoder.blks)):
#     print("---------------block %d-------------" % i)
#     decoder_blk = decoder.blks[i]
#     decoder_blk_ = decoder_.blks.modules[i]
#     # debug
#     # self.X = None
#     # self.enc_outputs = None
#     # self.enc_valid_lens = None
#     # self.X2 = None
#     # self.Y = None
#     # self.Y2 = None
#     # self.Z = None
#     # self.key_values = None
#     # self.dec_valid_lens = None
#     print("X: ", np.linalg.norm(
#         decoder_blk.X.detach().numpy() - decoder_blk_.X.numpy()))
#     print("enc_outputs: ", np.linalg.norm(
#         decoder_blk.enc_outputs.detach().numpy() - decoder_blk_.enc_outputs.numpy()))
#     print("enc_valid_lens: ", np.linalg.norm(
#         decoder_blk.enc_valid_lens.detach().numpy() - decoder_blk_.enc_valid_lens))
#     print("key_values: ", np.linalg.norm(
#         decoder_blk.key_values.detach().numpy() - decoder_blk_.key_values.numpy()))
#     if decoder_blk.dec_valid_lens is not None:
#         print("dec_valid_lens: ", np.linalg.norm(
#             decoder_blk.dec_valid_lens.detach().numpy() - decoder_blk_.dec_valid_lens))
#     print("=====attention1======")
#     print("queries: ", np.linalg.norm(
#         decoder_blk.attention1.X1.detach().numpy() - decoder_blk_.attention1.X1.numpy()))
#     print("keys: ", np.linalg.norm(
#         decoder_blk.attention1.X2.detach().numpy() - decoder_blk_.attention1.X2.numpy()))
#     print("values: ", np.linalg.norm(
#         decoder_blk.attention1.X3.detach().numpy() - decoder_blk_.attention1.X3.numpy()))
#     if decoder_blk.attention1.vl is not None:
#         print("vl: ", np.linalg.norm(
#             decoder_blk.attention1.vl.detach().numpy() - decoder_blk_.attention1.vl))
#     attention = d2l.DotProductAttention(dropout)
#     opt = attention(decoder_blk.attention1.X1, decoder_blk.attention1.X2, decoder_blk.attention1.X3, decoder_blk.attention1.vl)
#     attention_ = ndl.nn.DotProductAttention(dropout)
#     opt_ = attention_(decoder_blk_.attention1.X1, decoder_blk_.attention1.X2, decoder_blk_.attention1.X3, decoder_blk_.attention1.vl)
#     print("DotProductAttention: ", np.linalg.norm(
#         opt.detach().numpy() - opt_.numpy()))
#     print("output: ", np.linalg.norm(
#         decoder_blk.attention1.output.detach().numpy() - decoder_blk_.attention1.output.numpy()))
#     print("=====================")
#     print("X2: ", np.linalg.norm(
#         decoder_blk.X2.detach().numpy() - decoder_blk_.X2.numpy()))
#     print("Y: ", np.linalg.norm(
#         decoder_blk.Y.detach().numpy() - decoder_blk_.Y.numpy()))
#     print("Y2: ", np.linalg.norm(
#         decoder_blk.Y2.detach().numpy() - decoder_blk_.Y2.numpy()))
#     print("Z: ", np.linalg.norm(
#         decoder_blk.Z.detach().numpy() - decoder_blk_.Z.numpy()))
