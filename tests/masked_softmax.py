import torch
import numpy as np

X = np.random.randn(2, 2, 4)
print("X:",X)
valid_lens = np.array([2,3])
valid_lens = valid_lens.repeat(X.shape[1])
print("valid_lens", valid_lens)
X = X.reshape(-1, X.shape[-1])
print("X:", X)
maxlen = X.shape[-1]
mask = (torch.arange((maxlen), dtype=torch.float32)[None, :].numpy() < valid_lens[:, None])
mask_mul = mask.astype(np.float32)
mask_add = (~mask).astype(np.float32)*1e-6
print("mask:", mask)
print("mask_mul:", mask_mul)
print("mask_add:", mask_add)

y = X*mask_mul + mask_add
print("masked X:", y)