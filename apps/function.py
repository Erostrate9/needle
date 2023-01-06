import sys

sys.path.append('../python')
import needle as ndl

import math
import numpy as np
import torch
import torch.nn as nn
import d2l.torch as d2l

def set_attention(_attention, attention, use_bias=False):
    # _attention: d2l.MultiHeadAttention with PyTorch
    # attention: nn.MultiHeadAttention with Needle
    # multiHeadAttention_weight
    _attention.W_q.weight = torch.nn.Parameter(
        torch.tensor(attention.W_q.weight.numpy().T, dtype=torch.float32))
    _attention.W_k.weight = torch.nn.Parameter(
        torch.tensor(attention.W_k.weight.numpy().T, dtype=torch.float32))
    _attention.W_v.weight = torch.nn.Parameter(
        torch.tensor(attention.W_v.weight.numpy().T, dtype=torch.float32))
    _attention.W_o.weight = torch.nn.Parameter(
        torch.tensor(attention.W_o.weight.numpy().T, dtype=torch.float32))
    if use_bias:
        _attention.W_q.bias = torch.nn.Parameter(
            torch.tensor(attention.W_q.bias.numpy(), dtype=torch.float32))
        _attention.W_k.bias = torch.nn.Parameter(
            torch.tensor(attention.W_k.bias.numpy(), dtype=torch.float32))
        _attention.W_v.bias = torch.nn.Parameter(
            torch.tensor(attention.W_v.bias.numpy(), dtype=torch.float32))
        _attention.W_o.bias = torch.nn.Parameter(
            torch.tensor(attention.W_o.bias.numpy(), dtype=torch.float32))

def set_ffn(_ffn, ffn):
    _ffn.dense1.weight = torch.nn.Parameter(torch.tensor(ffn.dense1.weight.numpy().T, dtype=torch.float32))
    _ffn.dense1.bias = torch.nn.Parameter(torch.tensor(ffn.dense1.bias.numpy(), dtype=torch.float32))
    _ffn.dense2.weight = torch.nn.Parameter(torch.tensor(ffn.dense2.weight.numpy().T, dtype=torch.float32))
    _ffn.dense2.bias = torch.nn.Parameter(torch.tensor(ffn.dense2.bias.numpy(), dtype=torch.float32))

def set_encoder_blk(_encoder_blk, encoder_blk, use_bias=True):
    set_attention(_encoder_blk.attention, encoder_blk.attention, use_bias)
    # ffn_weight
    set_ffn(_encoder_blk.ffn, encoder_blk.ffn)

def set_encoder(_encoder, encoder, use_bias=False):
    # encoder: pytorch TransformerEncoder
    # encoder_: needle TransformerEncoder
    for i in range(len(_encoder.blks)):
        _encoder_blk = _encoder.blks[i]
        encoder_blk = encoder.blks.modules[i]
        # multiHeadAttention_weight
        set_encoder_blk(_encoder_blk, encoder_blk, use_bias=use_bias)
    _encoder.embedding.weight = torch.nn.Parameter(torch.tensor(encoder.embedding.weight.numpy(), dtype=torch.float32))

def set_decoder_blk(_decoder_blk, decoder_blk, use_bias):
    # multiHeadAttention_weight
    set_attention(_decoder_blk.attention1, decoder_blk.attention1, use_bias)
    set_attention(_decoder_blk.attention2, decoder_blk.attention2, use_bias)
    # ffn_weight
    set_ffn(_decoder_blk.ffn, decoder_blk.ffn)

def set_decoder(_decoder, decoder, use_bias=False):
    # decoder: pytorch TransformerDecoder
    # decoder_: needle TransformerDecoder
    for i in range(len(_decoder.blks)):
        _decoder_blk = _decoder.blks[i]
        decoder_blk = decoder.blks.modules[i]
        set_decoder_blk(_decoder_blk, decoder_blk, use_bias)
    _decoder.embedding.weight = torch.nn.Parameter(torch.tensor(decoder.embedding.weight.numpy(), dtype=torch.float32))
    _decoder.dense.weight = torch.nn.Parameter(torch.tensor(decoder.dense.weight.numpy().T, dtype=torch.float32))
    _decoder.dense.bias = torch.nn.Parameter(torch.tensor(decoder.dense.bias.numpy(), dtype=torch.float32))

def to_ndl(tensor, device=None, dtype="float32"):
    return ndl.Tensor(tensor.to(torch.device('cpu')).detach().numpy(), device=device, dtype=dtype)

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device, init='xavier'):
    """Train a model for sequence to sequence.

    Defined in :numref:`sec_seq2seq_decoder`"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    def kaiming_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.kaiming_uniform_(m._parameters[param])

    if init == 'xavier':
        net.apply(xavier_init_weights)
    elif init == 'kaiming':
        net.apply(kaiming_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = d2l.MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')