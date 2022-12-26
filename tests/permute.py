import torch
x = torch.randn(10,5,4,3)
print(x.shape)
axes = (1,3,2,0)
axes_ = [0]*len(axes)
for i, idx in enumerate(axes):
    axes_[idx] = i
print(x.permute(axes).shape)
print(x.permute(axes).permute(axes_).shape)