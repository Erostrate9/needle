"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * self.scalar * (power_scalar(a, self.scalar - 1)),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * lhs / (-rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / numpy.float32(self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(len(a.shape)))
        if self.axes is None:
            axes[-1], axes[-2] = axes[-2], axes[-1]
        else:
            axis1 = self.axes[0]
            axis2 = self.axes[1]
            axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return a.permute(tuple(axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = node.inputs[0].shape
        return (out_grad.reshape(shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        extra_len = len(out_grad.shape) - len(input_shape)
        index = tuple([i for i in reversed(range(len(out_grad.shape))) if (i < extra_len
                                                                           or out_grad.shape[i] != input_shape[
                                                                               i - extra_len])])
        return summation(out_grad, axes=index).reshape(input_shape)
        # in_shape = node.inputs[0].shape
        # out_shape = out_grad.shape
        # axes=[]
        # for i in range(-1,-len(in_shape)-1,-1):
        #   if(out_shape[i]!=in_shape[i]):
        #     axes.insert(0,len(out_shape)+i)
        # if len(out_shape)>len(in_shape):
        #   axes=list(range(len(out_shape)-len(in_shape)))+axes
        # return summation(out_grad, tuple(axes)).reshape(in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (tuple, list)):
            for i in self.axes:
                a = a.sum(i)
            return a
        else:
            return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = tuple(node.inputs[0].shape)
        out_shape = list(in_shape)
        if self.axes is None:
            out_shape = [1 for i in range(len(out_shape))]
        elif isinstance(self.axes, int):
            out_shape[self.axes] = 1
        else:
            for axis in self.axes:
                if axis < len(out_shape):
                    out_shape[axis] = 1
        return out_grad.reshape(out_shape).broadcast_to(in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        dshape = len(lhs.shape) - len(rhs.shape)
        small, big = lhs, rhs
        lhs_axes = ()
        rhs_axes = ()
        if (dshape > 0):
            big = lhs
            small = rhs
        elif (dshape < 0):
            big = rhs
            small = lhs
        # lhs.shape > rhs.shape
        if (dshape != 0 and len(big.shape) > 2):
            big_axes = ()
            small_axes = []
            for i in range(-3, -len(small.shape) - 1, -1):
                if small.shape[i] != big.shape[i]:
                    small_axes.insert(0, len(big.shape) + i)
            small_axes = list(range(abs(dshape))) + small_axes
            small_axes = tuple(small_axes)
            if dshape > 0:
                rhs_axes = small_axes
            else:
                lhs_axes = small_axes
        elif dshape == 0 and len(lhs.shape) > 2:
            # rhs.shape == lhs.shape
            axes = []
            for i in range(len(lhs.shape) - 2):
                if lhs.shape[i] != rhs.shape[i]:
                    axes.append(i)
                    if lhs.shape[i] == 1:
                        lhs_axes = axes
                        rhs_axes = ()
                    else:
                        lhs_axes = ()
                        rhs_axes = axes
            lhs_axes = tuple(lhs_axes)
            rhs_axes = tuple(rhs_axes)
        lhs_grad = summation(out_grad @ rhs.transpose(), lhs_axes)
        rhs_grad = summation(lhs.transpose() @ out_grad, rhs_axes)
        lhs_grad = lhs_grad if lhs_grad.shape == lhs.shape else lhs_grad.reshape(lhs.shape)
        rhs_grad = rhs_grad if rhs_grad.shape == rhs.shape else rhs_grad.reshape(rhs.shape)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ones = init.ones_like(node.inputs[0])
        return out_grad * (ones / node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor(node.numpy() > 0, device=node.device, dtype="float32") * out_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        y = array_api.log(array_api.exp(Z - array_api.broadcast_to(max_z, Z.shape)).sum(axis=self.axes, keepdims=True))
        return (y + array_api.broadcast_to(max_z, y.shape)).sum(axis=self.axes, keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0].cached_data
        Z_max = input.max(self.axes, keepdims=True)
        shape = list(input.shape)
        if self.axes is None:
            shape = [1 for _ in range(len(input.shape))]
        elif isinstance(self.axes, (list, tuple)):
            for i in self.axes:
                shape[i] = 1
        else:
            shape[self.axes] = 1

        out_grad = out_grad.reshape(shape).broadcast_to(input.shape)
        input -= Z_max.broadcast_to(input.shape)
        expz = array_api.exp(input)
        return out_grad / Tensor(expz.sum(self.axes, keepdims=True).broadcast_to(input.shape) / expz,
                                 device=out_grad.device)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        return (init.ones_like(Z, requires_grad=False) - tanh(Z) * tanh(Z)) * out_grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        return array_api.stack(args, axis=self.axis)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # (a,) = node.inputs
        # assert isinstance(a, TensorTuple)
        # assert isinstance(out_grad, Tensor)

        return out_grad.split(self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        return array_api.split(A, axis=self.axis)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        assert isinstance(a, Tensor)
        assert isinstance(out_grad, TensorTuple)
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis < a.ndim:
                new_shape[axis] *= (self.dilation + 1)
        out = array_api.full(tuple(new_shape), 0, dtype=a.dtype, device=a.device)
        sl = [slice(None)] * a.ndim
        for axis in self.axes:
            if axis < a.ndim:
                sl[axis] = slice(0, new_shape[axis], self.dilation + 1)
        out[tuple(sl)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        sl = [slice(None)] * a.ndim
        for axis in self.axes:
            if axis < a.ndim:
                sl[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(sl)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        """
        im2col method for convolution
        stride: 1 as default
        padding: padding applied to the spatial dimensions (i.e., axes 1 and 2), 0 as default
        """
        self.stride = stride
        self.padding = padding

    def compute(self, Z, weight):
        """
        Compute im2col on Z with weight as its kernel.
        Parameters:
        Z - N * H * W * C_in NDArray
        W - K * K * C_in * C_out NDArray
        """
        ### BEGIN YOUR SOLUTION
        # Z = Z.compact()
        weight = weight.compact()
        # padding
        Z = Z.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        # im2col
        N, H, W, C_in = Z.shape
        K, _, _, C_out = weight.shape
        Ns, Hs, Ws, Cs = Z.strides

        inner_dim = K * K * C_in
        new_H = (H - K) // self.stride + 1
        new_W = (W - K) // self.stride + 1
        Z_ = Z.as_strided(shape=(N, new_H, new_W, K, K, C_in),
                          strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact() \
              .reshape((-1, inner_dim))
        out = Z_ @ weight.reshape((-1, C_out))
        return out.reshape((N, new_H, new_W, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Z_grad    (N, H, W, C_in)
        # weight_grad    (K, K, C_in, C_out)
        # out_grad   (N, (H-K+2P)//S+1, (W-K+2P)//S+1, C_out)

        Z, weight = node.inputs
        K, _, _, C_out = weight.shape

        # X.grad = out_grad @ W.transpose
        # W.grad = X.transpose @ out_grad
        # X.grad = ≈conv(≈out_grad, ≈W)
        # W.grad = ≈conv(≈X, ≈out_grad)
        # the transpose of a convolution can be found by simply flipping the kernel.
        weight = weight.flip((0, 1)).transpose((2,3))
        # stride
        if self.stride > 1:
            out_grad = out_grad.dilate((1, 2), self.stride-1)
        # (N, newH, newW, C_out) =>  (N, H, W, C_out)
        # (H-K+2P) + 1 - K + 2x + 1 == H
        # => x = K - P -1
        Z_grad = conv(out_grad, weight, 1, K - self.padding - 1)

        # (N, H, W, C_in) (N, (H-K+2P)//S+1, (W-K+2P)//S+1, C_out)
        # (K, K, C_in, C_out)
        # (C_in, W, H, N) conv (newW, newH, N, C_out) => (C_in, K, K, C_out)
        out_grad = out_grad.transpose((0,2))
        Z = Z.transpose((0,3)).transpose((1,2))
        # H  - (H-K+2P+1) + 2x +1 == K
        # => x = P
        weight_grad = conv(Z, out_grad, 1, self.padding)
        weight_grad = weight_grad.transpose((0,2))
        return Z_grad, weight_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
