from __future__ import print_function
import torch

x = torch.ones(2, 2, requires_grad=True) # this specifies that this tensor should have pytorch's autograd run for it. This makes the tensor's operations be tracked for use in backpropogation.
print(x)
y = x + 2
print(y)
print(y.grad_fn) #since y was created as a result of an operation, it has a gradient flag. this line prints whether a tensor has a flag or not (which in this case, it does)

z = y * y * 3
out = z.mean()

print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))

print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)

b = (a * a).sum() # the .sum() is either a numpy thing or just a python thing idk which but I imagine it gets the sum of (a* a)
print(b.grad_fn)
# you can retroactively change the flag that states if a tensor requires gradient descent.

out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)


y = x * 2

while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)