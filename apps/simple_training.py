import sys

sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time
import numpy as np

device = ndl.cpu()


### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    total_loss = 0
    total_examples = len(dataloader.dataset)
    total_corrects = 0
    loss_fn = nn.SoftmaxLoss()
    for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.detach().numpy() * y.shape[0]
        y_hat = np.argmax(logits.detach().numpy(), axis=1)
        total_corrects += np.sum(y_hat == y.numpy())
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    avg_loss = total_loss / total_examples
    avg_acc = total_corrects / total_examples
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if optimizer == ndl.optim.SGD:
        opt = optimizer(params=model.parameters(), lr=lr, momentum=0.875, weight_decay=weight_decay)
    else:
        opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        start_time = time.time()
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model,
                                                  loss_fn=loss_fn(), opt=opt)
        end_time = time.time()
        print("train epoch{}: avg_acc: {}, avg_loss: {}, time cost: {}".format(
            i, avg_acc, avg_loss, end_time - start_time))
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model,
                                              loss_fn=loss_fn(), opt=None)
    print("evaluate: avg_acc: {}, avg_loss: {}".format(avg_acc, avg_loss))
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
                      clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    total_loss = 0
    total_examples = 0
    total_corrects = 0
    nbatch, batch_size = data.shape
    h = None
    for i in range(0, nbatch-1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        b_len = y.shape[0]
        total_examples += b_len
        out, h = model(X, h)
        # must detach the h to avoid augment the computational graph
        if isinstance(h, tuple):
            h = tuple(_.detach() for _ in h)
        else:
            h = h.detach()
        # out (seq_len*bs, output_size)
        # y = (bptt*bs,)
        loss = loss_fn(out, y)
        total_loss += loss.detach().numpy().squeeze() * b_len
        y_hat = np.argmax(out.detach().numpy(), axis=1)
        total_corrects += np.sum(y_hat == y.numpy())
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

            if clip is not None:
                for param in opt.params:
                    param.grad = ndl.Tensor(
                        clip * param.grad.data / np.linalg.norm(param.grad.data),
                        device=param.device,
                        dtype=param.dtype,
                    )
    avg_loss = total_loss / total_examples
    avg_acc = total_corrects / total_examples
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
              lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
              device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        start_time = time.time()
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=opt,
                      clip=clip, device=device, dtype=dtype)
        end_time = time.time()
        print("train epoch{}: avg_acc: {}, avg_loss: {}, time cost: {}".format(
            i, avg_acc, avg_loss, end_time - start_time))
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
                 device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(), opt=None,
                      clip=None, device=device, dtype=dtype)
    print("evaluate: avg_acc: {}, avg_loss: {}".format(avg_acc, avg_loss))
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    # dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    # dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    # model = ResNet9(device=device, dtype="float32")
    # train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
