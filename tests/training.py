### import required packages
import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l
import math
import apps.function as F

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""

    optimizer = ndl.optim.Adam(net.parameters(), lr=lr)
    loss = ndl.nn.MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.reset_grad()
            X, X_valid_len, Y, Y_valid_len = [x for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0]).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            X, dec_input = F.to_ndl(X, device), F.to_ndl(dec_input, device)
            X_valid_len = X_valid_len.detach().to(torch.device('cpu')).numpy()
            Y = F.to_ndl(Y, device)
            Y_valid_len = Y_valid_len.detach().to(torch.device('cpu')).numpy()

            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`

            num_tokens = Y_valid_len.sum()

            optimizer.step()

            metric.add(l.numpy().sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

# Training process using PyTorch's implementation. [2]
use_bias = True
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, _device = 0.005, 200, d2l.try_gpu()
device = ndl.cuda()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = ndl.nn.TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout, use_bias=use_bias, device=device)

decoder = ndl.nn.TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout, use_bias=use_bias, device=device)


net = ndl.nn.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)