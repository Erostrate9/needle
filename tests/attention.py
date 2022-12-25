import sys
sys.path.append('../python')
sys.path.append('../apps')
import needle as ndl
import numpy as np
import torch
import torch.nn as nn

def softmax(Z):
    Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return Z / Z.sum(axis=-1, keepdims=True)
def multihead_attention(X, mask, heads, W_KQV, W_out):
    B, T, d = X.shape
    K, Q, V = np.split(X@W_KQV, 3, axis=-1)
    # B x T x d =>
    # B x heads x T x d/heads
    K, Q, V = [a.reshape(B, T, heads, d//heads).swapaxes(1, 2) for a in (K, Q, V)]
    attn = softmax(K@Q.swapaxes(-1,-2) / np.sqrt(d // heads) + mask)
    return (attn @ V).swapaxes(1,2).reshape(B, T, d) @ W_out, attn
def layer_norm(Z, eps):
    return (Z - Z.mean(axis=-1, keepdims= True)) / np.sqrt(Z.var(axis=-1, keepdims=True) + eps)


def relu(Z):
    return np.maximum(Z, 0)
def transformer(X, mask, heads, W_KQV, W_out, W_ff1, W_ff2, eps):
    Z = layer_norm(X + multihead_attention(X,mask,heads,W_KQV,W_out)[0], eps)
    return layer_norm(Z + relu(Z@W_ff1)@W_ff2, eps)

B, T, d = 50, 100, 64
X = torch.randn(B, T, d, dtype=torch.float32)
M = torch.triu(-float("inf") * torch.ones(T,T), 1)

heads = 4
attn = nn.MultiheadAttention(d, heads, bias = False, batch_first= True)

mask = ndl.Tensor(M.numpy(), device=ndl.cpu())
W_KQV = ndl.Tensor(attn.in_proj_weight.detach().numpy().T, device=ndl.cpu())
W_out = ndl.Tensor(attn.out_proj.weight.detach().numpy().T, device=ndl.cpu())
X_ = ndl.Tensor(X.numpy(), device=ndl.cpu())

Y_, A_ = attn(X, X, X, attn_mask = M)
Y, A = ndl.nn.MultiheadAttention(mask, heads, W_KQV, W_out,
                                 device=ndl.cpu())(X_)

print(np.linalg.norm(Y.numpy()-Y_.detach().numpy()))

trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128,
                                   dropout=0.0, batch_first=True)
trans.linear1.bias.data.zero_()
trans.linear2.bias.data.zero_()

eps = 1e-5
attn_output = multihead_attention(X.numpy(),M.numpy(),heads,
               trans.self_attn.in_proj_weight.detach().numpy().T,
               trans.self_attn.out_proj.weight.detach().numpy().T,)[0]
i = X + attn_output
Z = layer_norm(i, eps)
Z_ = ndl.nn.LayerNorm(eps=eps)(ndl.Tensor(i.detach().numpy(), device=ndl.cpu(), dtype="float32"))

print("norm:", np.linalg.norm(Z_.numpy()-Z.numpy()))
B, T, d = 50, 100, 64
X = torch.randn(B, T, d)
M = torch.triu(-float("inf") * torch.ones(T,T), 1)



Y_ = trans(X,M)
mask = ndl.Tensor(M.numpy(), device=ndl.cpu(), dtype="float32")
W_KQV = ndl.Tensor(trans.self_attn.in_proj_weight.detach().numpy().T, device=ndl.cpu(), dtype="float32")
W_out = ndl.Tensor(trans.self_attn.out_proj.weight.detach().numpy().T, device=ndl.cpu(), dtype="float32")
W_ff1 = ndl.Tensor(trans.linear1.weight.detach().numpy().T, device=ndl.cpu(), dtype="float32")
W_ff2 = ndl.Tensor(trans.linear2.weight.detach().numpy().T, device=ndl.cpu(), dtype="float32")
X_ = ndl.Tensor(X.numpy(), device=ndl.cpu(), dtype="float32")
Y = ndl.nn.Transformer(mask, heads,
                W_KQV, W_out, W_ff1, W_ff2,
                eps=eps)(X_)
print("Transformer:", np.linalg.norm(Y.numpy()-Y_.detach().numpy()))