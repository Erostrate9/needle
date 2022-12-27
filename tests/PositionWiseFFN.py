import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l

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

ffn_ = ndl.nn.PositionWiseFFN(4, 4, 8, device=ndl.cpu())
one = ndl.init.ones(2, 3, 4 , device=ndl.cpu(), dtype="float32")
y_ = ffn_(one)
print(y_.shape)
w1 = torch.tensor(ffn_.dense1.weight.numpy())
b1 = torch.tensor(ffn_.dense1.bias.numpy())
w2 = torch.tensor(ffn_.dense2.weight.numpy())
b2 = torch.tensor(ffn_.dense2.bias.numpy())
ffn = PositionWiseFFN(4, 4, 8)
ffn.dense1.weight = torch.nn.Parameter(torch.tensor(ffn_.dense1.weight.numpy().T, dtype=torch.float32))
ffn.dense1.bias = torch.nn.Parameter(torch.tensor(ffn_.dense1.bias.numpy(), dtype=torch.float32))
ffn.dense2.weight = torch.nn.Parameter(torch.tensor(ffn_.dense2.weight.numpy().T, dtype=torch.float32))
ffn.dense2.bias = torch.nn.Parameter(torch.tensor(ffn_.dense2.bias.numpy(), dtype=torch.float32))
ffn.eval()
y = ffn(torch.ones((2, 3, 4)))
print(y.shape)
print("PositionWiseFFN", np.linalg.norm(y_.numpy()-y.detach().numpy()))