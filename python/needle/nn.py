"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math
import torch

import operator
from functools import reduce


def prod(x):
    return reduce(operator.mul, x, 1)


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
            self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.device = device
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype
        )
        self.bias = Parameter(
            init.kaiming_uniform(out_features, 1).reshape((1, out_features)), device=device, dtype=dtype
        ) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = X @ self.weight
        if self.bias: res += self.bias.broadcast_to(res.shape)
        return res
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # x: a tensor of shape (B,X_0,X_1,...)
        # return: x flattened to the shape of (B, X_0 * X_1 * ...)
        B = X.shape[0]
        k = np.prod(list(X.shape[1:]))
        return X.reshape((B, k))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        one = init.ones_like(x, device=x.device)
        return one / (one + ops.exp(-x))
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def add_module(self, module):
        self.modules.append(module)

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # n: batch_num; k: class_num
        n, k, *_ = logits.shape
        y_one_hot = init.one_hot(k, y, device=y.device, dtype=y.dtype)
        # axes=(1,)
        logsumexp = ops.logsumexp(logits, axes=(1,))
        z_y = (logits * y_one_hot).sum(axes=(1,))
        return (logsumexp - z_y).sum() / n
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True), device=device,
                                dtype=dtype)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True), device=device,
                              dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, dim = x.shape
        assert dim == self.dim
        e, var, y = None, None, None
        weight = self.weight.reshape((1, dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, dim)).broadcast_to(x.shape)
        if (self.training):
            e = x.sum(axes=(0,)) / n
            self.running_mean.data = ((1 - self.momentum) * self.running_mean + self.momentum * e).data
            e = e.reshape((1, dim)).broadcast_to(x.shape)
            var = ((x - e) ** 2).sum(axes=(0,)) / n
            self.running_var.data = ((1 - self.momentum) * self.running_var + self.momentum * var).data
            var = var.reshape((1, dim)).broadcast_to(x.shape)
        else:
            e = self.running_mean.reshape((1, dim)).broadcast_to(x.shape).data
            var = self.running_var.reshape((1, dim)).broadcast_to(x.shape).data

        norm = ((x - e) / ((var + self.eps) ** 0.5))
        y = weight * norm + bias
        if (self.training == False):
            y = y.data
        return y
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: (n x dim)
        # e: (1 x dim)
        # var:(1 x dim)
        n, dim = x.shape
        assert dim == self.dim
        e = (x.sum(axes=(1,)) / dim).reshape((n, 1)).broadcast_to((n, dim))
        var = (((x - e) ** 2).sum(axes=(1,)) / dim).reshape((n, 1)).broadcast_to((n, dim))
        weight = self.weight.reshape((1, dim)).broadcast_to((n, dim))
        bias = self.bias.reshape((1, dim)).broadcast_to((n, dim))
        y = weight * (x - e) / ((var + self.eps) ** 0.5) + bias
        return y
        ### END YOUR SOLUTION


class LayerNorm(Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, Z: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Z: (n1 x n2 x ... x nd)
        # e: (n1 x n2 x ... x nd)
        # var:(n1 x n2 x ... x nd)
        axis = len(Z.shape) - 1
        d = Z.shape[-1]
        # mean
        e = Z.sum(axes=(axis,)).reshape(Z.shape[:-1] + (1,)) / d
        # var
        var = (((Z - e.broadcast_to(Z.shape)) ** 2).sum(axes=(axis,)) / d).reshape(Z.shape[:-1] + (1,))

        a = (Z - e.broadcast_to(Z.shape))
        b = ((var.broadcast_to(Z.shape) + self.eps) ** 0.5)
        return a / b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        if self.training:
            # probi = 0 with probability p
            prob = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
            y = x / (1 - self.p) * prob
        return y
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same, stride=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        fan_in = kernel_size * kernel_size * in_channels
        fan_out = kernel_size * kernel_size * out_channels
        self.weight = Parameter(init.kaiming_uniform(fan_in, fan_out,
                                                     shape=weight_shape, device=device, dtype=dtype))
        bias_bound = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
        self.bias = Parameter(init.rand(out_channels, low=-bias_bound,
                                        high=bias_bound, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # inputs in NCHW format: NCHW -> NHWC
        x = ops.transpose(ops.transpose(x, (1, 2)), (2, 3))
        # H+2P-K+1 == H => P = K//2
        padding = self.kernel_size // 2
        # NHWC
        convolution = ops.conv(x, self.weight, stride=self.stride, padding=padding)
        if self.bias:
            bias = self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(convolution.shape)
            convolution += bias
        # NHWC -> NCHW
        out = ops.transpose(ops.transpose(convolution, (2, 3)), (1, 2))
        return out
        ### END YOUR SOLUTION


class ConvBN(Module):
    def __init__(self, a, b, k, s, device=None, dtype="float32"):
        super().__init__()
        conv2d = Conv(a, b, k, stride=s, bias=True, device=device, dtype=dtype)
        self.convbn = Sequential(
            conv2d,
            BatchNorm2d(b, device=device, dtype=dtype),
            ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convbn(x)


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Use split & stack instead of getitem & setitem
        if nonlinearity == 'relu':
            self.act_fn = ReLU()
        elif nonlinearity == 'tanh':
            self.act_fn = Tanh()
        else:
            self.act_fn = None
            raise ValueError("Only support tanh & relu for RNNCell.")
        sqrt_k = (1.0 / hidden_size) ** 0.5
        self.device = device
        self.dtype = dtype
        self.W_ih = Parameter(init.rand(input_size, hidden_size,
                                        low=-sqrt_k, high=sqrt_k,
                                        device=device, dtype=dtype,
                                        requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size,
                                        low=-sqrt_k, high=sqrt_k,
                                        device=device, dtype=dtype,
                                        requires_grad=True))
        self.bias_ih = Parameter(init.rand(hidden_size,
                                           low=-sqrt_k, high=sqrt_k,
                                           device=device, dtype=dtype,
                                           requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size,
                                           low=-sqrt_k, high=sqrt_k,
                                           device=device, dtype=dtype,
                                           requires_grad=True)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape
        _, hidden_size = self.W_ih.shape
        assert input_size == _
        if h is None:
            h = init.zeros(bs, hidden_size, device=self.device, dtype=self.dtype)
        if self.bias:
            h_ = X @ self.W_ih + \
                 self.bias_ih.reshape((1, hidden_size)).broadcast_to((bs, hidden_size)) + \
                 h @ self.W_hh + \
                 self.bias_hh.reshape((1, hidden_size)).broadcast_to((bs, hidden_size))
        else:
            h_ = X @ self.W_ih + h @ self.W_hh
        return self.act_fn(h_)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None,
                 dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias,
                                  nonlinearity=nonlinearity, device=device, dtype=dtype)]
        for i in range(1, num_layers):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias=bias,
                                          nonlinearity=nonlinearity, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h_0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h_0 is None:
            h_0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        num_layers, bs, hidden_size = h_0.shape
        assert hidden_size == self.hidden_size and num_layers == self.num_layers and input_size == self.input_size

        X = ops.split(X, 0)  # TensorTuple
        h_0 = ops.split(h_0, 0)  # TensorTuple
        h = [h_0[i] for i in range(num_layers)]
        output = []  # hidden state in the last layer
        for t in range(seq_len):
            h_l_t = self.rnn_cells[0](X[t], h[0])
            h[0] = h_l_t
            for l in range(1, num_layers):
                h_l_t = self.rnn_cells[l](h_l_t, h[l])
                h[l] = h_l_t
            output.append(h_l_t)
        output = ops.stack(output, 0)
        h_n_tuple = ops.make_tuple(*(h[layer] for layer in range(num_layers)))
        h_n = ops.stack(h_n_tuple, 0)
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        sqrt_k = (1.0 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size,
                                        low=-sqrt_k, high=sqrt_k,
                                        device=device, dtype=dtype,
                                        requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size,
                                        low=-sqrt_k, high=sqrt_k,
                                        device=device, dtype=dtype,
                                        requires_grad=True))
        self.bias_ih = Parameter(init.rand(4 * hidden_size,
                                           low=-sqrt_k, high=sqrt_k,
                                           device=device, dtype=dtype,
                                           requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(4 * hidden_size,
                                           low=-sqrt_k, high=sqrt_k,
                                           device=device, dtype=dtype,
                                           requires_grad=True)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (bs, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape
        h0, c0 = (init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype),
                  init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
                  ) if h is None else (h[0], h[1])
        _, hidden_size = h0.shape
        assert bs == _ and input_size == self.input_size and hidden_size == self.hidden_size
        ifgo = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            bias_ih = self.bias_ih.reshape((1, self.bias_ih.shape[0])).broadcast_to((bs, 4 * hidden_size))
            bias_hh = self.bias_hh.reshape((1, self.bias_hh.shape[0])).broadcast_to((bs, 4 * hidden_size))
            ifgo = ifgo + bias_ih + bias_hh
        sigmoid = Sigmoid()
        # ifgo: bs * 4*hidden_size
        ifgo = ifgo.reshape((bs, 4, hidden_size)).split(1)  # TupleTensor
        i, f, g, o = ifgo[0], ifgo[1], ifgo[2], ifgo[3]
        i, f, g, o = sigmoid(i), sigmoid(f), ops.tanh(g), sigmoid(o)
        c_ = f * c0 + i * g
        h_ = o * ops.tanh(c_)
        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype)]
        for i in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        h_0, c_0 = (h[0], h[1]) if h is not None else (
            init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype),
            init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        )
        num_layers, bs, hidden_size = h_0.shape
        assert hidden_size == self.hidden_size and num_layers == self.num_layers and input_size == self.input_size
        X = ops.split(X, 0)  # TensorTuple
        h_0 = ops.split(h_0, 0)  # TensorTuple
        c_0 = ops.split(c_0, 0)  # TensorTuple
        h = [h_0[i] for i in range(num_layers)]
        c = [c_0[i] for i in range(num_layers)]
        output = []  # hidden state in the last layer
        ###
        for t in range(seq_len):
            h_l_t, c_l_t = self.lstm_cells[0](X[t], (h[0], c[0]))
            h[0] = h_l_t
            c[0] = c_l_t
            for l in range(1, num_layers):
                h_l_t, c_l_t = self.lstm_cells[l](h_l_t, (h[l], c[l]))
                h[l] = h_l_t
                c[l] = c_l_t
            output.append(h_l_t)
        output = ops.stack(output, 0)
        h_n_tuple = ops.make_tuple(*(h[layer] for layer in range(num_layers)))
        c_n_tuple = ops.make_tuple(*(c[layer] for layer in range(num_layers)))
        h_n = ops.stack(h_n_tuple, 0)
        c_n = ops.stack(c_n_tuple, 0)
        ###
        return output, (h_n, c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        # (seq_len, bs, num_embeddings)
        one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype)
        seq_len, bs, num_embeddings = one_hot.shape
        one_hot = one_hot.reshape((seq_len * bs, num_embeddings))
        # (seq_len, bs, num_embeddings) @ (num_embeddings, embedding_dim)
        return (one_hot @ self.weight).reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION


class Encoder(Module):
    """The base encoder interface for the encoder-decoder architecture."""

    def __init__(self):
        super().__init__()

        # Later there can be additional arguments (e.g., length excluding padding)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(Module):
    """The base decoder interface for the encoder-decoder architecture."""

    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(Module):
    """The base class for the encoder-decoder architecture."""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class MultiheadAttention_test(Module):
    def __init__(self, mask, heads, W_KQV, W_out, device=None, dtype="float32"):
        super().__init__()
        self.mask = mask
        self.heads = heads
        self.W_KQV = Parameter(W_KQV, device=device, dtype=dtype,
                               requires_grad=True)
        self.W_out = Parameter(W_out, device=device, dtype=dtype,
                               requires_grad=True)
        self.attention_weights = None

    def forward(self, X: Tensor) -> Tensor:
        def get_tensors(ttuple, start, end):
            res = []
            for i in range(start, end):
                if i >= len(ttuple):
                    break
                res.append(ttuple[i])
            return ops.MakeTensorTuple()(*res)

        B, T, d = X.shape
        res = (X.reshape((B * T, d)) @ self.W_KQV).reshape((B, T, self.W_KQV.shape[-1]))

        KQV = ops.split(res, len(X.shape) - 1)
        n = self.W_KQV.shape[-1] // 3

        K = ops.stack(get_tensors(KQV, 0, n), len(X.shape) - 1).reshape((B, T, self.heads, d // self.heads)).transpose(
            (1, 2))
        Q = ops.stack(get_tensors(KQV, n, n * 2), len(X.shape) - 1).reshape(
            (B, T, self.heads, d // self.heads)).transpose((1, 2))
        V = ops.stack(get_tensors(KQV, n * 2, n * 3), len(X.shape) - 1).reshape(
            (B, T, self.heads, d // self.heads)).transpose((1, 2))
        # B x T x d =>
        # B x heads x T x d/heads
        # K@Q.T: B x heads x T x T
        # mask: T x T
        attn = ops.softmax(ops.batch_matmul(K, Q.transpose()) / ((d // self.heads) ** 0.5) + self.mask.broadcast_to(
            (B, self.heads, T, T)))
        self.attention_weights = attn
        attn_output = (ops.batch_matmul(attn, V).transpose((1, 2)).reshape((B * T, d)) @ self.W_out).reshape(
            (B, T, self.W_out.shape[-1]))
        return attn_output, attn


class Transformer_test(Module):
    def __init__(self, mask, heads, W_KQV, W_out, W_ff1, W_ff2, eps, device=None, dtype="float32"):
        super().__init__()
        self.mask = mask
        self.heads = heads
        self.W_KQV = Parameter(W_KQV, device=device, dtype=dtype,
                               requires_grad=True)
        self.W_out = Parameter(W_out, device=device, dtype=dtype,
                               requires_grad=True)
        self.W_ff1 = Parameter(W_ff1, device=device, dtype=dtype,
                               requires_grad=True)
        self.W_ff2 = Parameter(W_ff2, device=device, dtype=dtype,
                               requires_grad=True)
        self.layer_norm = LayerNorm(eps)
        self.multihead_attention = MultiheadAttention_test(mask, heads, W_KQV, W_out, device=device, dtype=dtype)
        self.relu = ReLU()

    def forward(self, X: Tensor) -> Tensor:
        Z = self.layer_norm(X + self.multihead_attention(X)[0])
        B, T, d = Z.shape
        rhs = (self.relu(Z.reshape((B * T, d)) @ self.W_ff1) @ self.W_ff2).reshape((B, T, d))
        return self.layer_norm(Z + rhs)


class DotProductAttention(Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_batch_dot`"""

    def __init__(self, dropout, num_heads=None):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.num_heads = num_heads  # To be covered later

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def masked_softmax(self, X, valid_lens):
        # valid_lens: numpy array
        def _sequence_mask(X: Tensor, valid_lens, value=0):
            # X: n * d
            maxlen = X.shape[-1]
            mask = (torch.arange((maxlen), dtype=torch.float32)[None, :].numpy() < valid_lens[:, None])
            mask_mul = mask.astype(np.float32)
            mask_add = (~mask).astype(np.float32) * value
            mask_mul = Tensor(mask_mul, device=X.device, dtype=X.dtype)
            mask_add = Tensor(mask_add, device=X.device, dtype=X.dtype)
            return X * mask_mul + mask_add

        if valid_lens is None:
            return ops.softmax(X)
        else:
            shape = X.shape
            if len(valid_lens.shape) == 1:
                assert valid_lens.shape[0] == X.shape[0]
                valid_lens = valid_lens.repeat(X.shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            # On the last axis, replace masked elements with a very large negative
            # value, whose exponentiation outputs 0
            X = _sequence_mask(X.reshape((prod(shape[:-1]), shape[-1])), valid_lens, value=-1e6)
            return ops.softmax(X.reshape(shape))

    def forward(self, queries, keys, values, valid_lens=None,
                window_mask=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = ops.batch_matmul(queries, keys.transpose((1, 2))) / math.sqrt(d)
        if window_mask is not None:  # To be covered later
            num_windows = window_mask.shape[0]
            n, num_queries, num_kv_pairs = scores.shape
            # Shape of window_mask: (num_windows, no. of queries,
            # no. of key-value pairs)
            scores = ops.reshape(
                scores, (n // (num_windows * self.num_heads), num_windows,
                         self.num_heads, num_queries, num_kv_pairs
                         )) + ops.reshape(window_mask, (1, window_mask.shape[0], 1) + window_mask.shape[1:])
            scores = ops.reshape(scores, (n, num_queries, num_kv_pairs))
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        return ops.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, device=None, dtype="float32"):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = Linear(query_size, num_hiddens, bias=bias, device=device, dtype=dtype)
        self.W_k = Linear(key_size, num_hiddens, bias=bias, device=device, dtype=dtype)
        self.W_v = Linear(value_size, num_hiddens, bias=bias, device=device, dtype=dtype)
        self.W_o = Linear(num_hiddens, num_hiddens, bias=bias, device=device, dtype=dtype)
        ### test
        self.vl = None
        self.output = None

    def forward(self, queries, keys, values, valid_lens, window_mask=None):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = valid_lens.repeat(repeats=self.num_heads, axis=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        self.vl = valid_lens
        output = self.attention(queries, keys, values, valid_lens,
                                window_mask)
        self.output = output
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads.

        Defined in :numref:`sec_multihead-attention`"""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1))
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute((0, 2, 1, 3))
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.reshape((-1, X.shape[2], X.shape[3]))
        return X

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv.

        Defined in :numref:`sec_multihead-attention`"""
        X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2]))
        X = X.permute((0, 2, 1, 3))
        return X.reshape((X.shape[0], X.shape[1], -1))


class PositionWiseFFN(Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, device=None, dtype="float32"):
        super().__init__()
        self.dense1 = Linear(ffn_num_input, ffn_num_hiddens, device=device, dtype=dtype)
        self.relu = ReLU()
        self.dense2 = Linear(ffn_num_hiddens, ffn_num_outputs, device=device, dtype=dtype)

    def forward(self, X: Tensor) -> Tensor:
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.ln = LayerNorm(eps=1e-5)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        # ###
        # import torch.nn as nn
        # import torch
        # ln = nn.LayerNorm(X.shape[-1])
        # x = self.dropout(Y) + X
        # y_ = ln(torch.tensor(x.numpy()))
        # y = self.ln(x)
        # print("AddNorm", np.linalg.norm(y.numpy()-y_.detach().numpy()))
        # ###
        return self.ln(self.dropout(Y) + X)


class TransformerEncoderBlock_test(Module):
    """Transformer Encoder Block"""

    def __init__(self, mask, num_heads, W_KQV, W_out,
                 ffn_num_input, ffn_num_hiddens, num_hiddens,
                 eps, dropout, device=None, dtype="float32"):
        super().__init__()
        self.attention = MultiheadAttention_test(mask, num_heads, W_KQV, W_out, device=device, dtype=dtype)
        self.addnorm1 = AddNorm(eps, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens, device=device, dtype=dtype)
        self.addnorm2 = AddNorm(eps, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X)[0])
        return self.addnorm2(Y, self.ffn(Y))


class PositionalEncoding(Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        P = Tensor(self.P[:, :X.shape[1], :], device=X.device, dtype=X.dtype, requires_grad=False)
        X = X + P
        return self.dropout(X)


class TransformerEncoder_test(Encoder):
    """Transformer Encoder"""

    def __init__(self, vocab_size, num_hiddens, num_blks, mask, num_heads, W_KQV, W_out,
                 ffn_num_input, ffn_num_hiddens,
                 eps, dropout, device=None, dtype="float32"):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = Embedding(vocab_size, num_hiddens, device=device, dtype=dtype)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = Sequential()
        for i in range(num_blks):
            self.blks.add_module(
                TransformerEncoderBlock_test(mask, num_heads, W_KQV, W_out,
                                             ffn_num_input, ffn_num_hiddens, num_hiddens,
                                             eps, dropout, device=device, dtype=dtype))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention_weights
        return X


class TransformerEncoderBlock(Module):
    """Transformer Encoder Block"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, device=None, dtype="float32"):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            bias=use_bias, device=device, dtype=dtype)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens, device=device, dtype=dtype)
        self.addnorm2 = AddNorm(dropout)
        ###
        # self.X = None
        # self.attn = None
        # self.Y = None
        # self.res = None
        # self.attn = None
        # self.ffnY = None

    def forward(self, X, valid_lens):
        # self.X = X.numpy()
        # attn = self.attention(X, X, X, valid_lens)
        # self.attn = attn.numpy()
        # Y = self.addnorm1(X, attn)
        # self.Y = Y.numpy()
        # ffnY = self.ffn(Y)
        # self.ffnY = ffnY.numpy()
        # res = self.addnorm2(Y, ffnY)
        # self.res = res.numpy()
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
