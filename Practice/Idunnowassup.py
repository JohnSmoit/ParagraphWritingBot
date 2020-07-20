from __future__ import print_function
import torch
import numpy as np

# by my understanding, tensors are essentially 
# arrays that can have multiple dimensions 
# and can have operations easily performed on them due to pytorch.
# I imagine that each dimension in a tensor represents a different layer of neurons in the neural network (generally speaking).
# in the examples below, there's not much going with regards to neural networks, just tensor manipulation and initialization.
# try commenting parts out and experimenting with the different tensor initializations.
# the class "torch" refers to the main pytorch class and anything following it is an initialization method for a different tensor.

# there is probably more to each member of torch than I say in my notes, I still don't really know what I'm talking about.

x = torch.empty(5, 3) # creates an "empty" tensor of the dimensions (x, y,...) specified. The values within will not be zeros or null, but rather very large numbers which are completely random within the bit limit.
print(x)
x = torch.rand(5, 3) # creates a tensor of the dimensions (x, y,...) specified with values within ranging from -1 to 1. This takes into account the datatype the tensor is storing. I think the default type is float / double
print(x)
x = torch.zeros(5, 3, dtype=torch.long) # creates a tensor filled entirely with zeroes
print(x)
x = torch.tensor([1000.2, 55.55]) # creates a tensor made up of specified values put in as arguments.
print(x)
x = x.new_ones(5, 3, dtype=torch.double) # creates a tensor of ones that are of the datatype (dtype) "double"
print(x)
x = torch.randn_like(x, dtype=torch.float)# creates a tensor that shares properties (from what I see so far, dimensions) of an input tensor full of random numbers with values the same as torch.rand that are of the datatype (dtype) "double"
print(x)
print(x.size())
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
y.add_(x)
print(y)
print(x[1, :])
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
a = torch.ones(5)

b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda") # cuda is a program / api that makes use of a computer's Graphical Processing Unit (GPU) to perform other calculations alongside (speculation) CPU to bolster performance of processing-heavy neural networks and likely other things.
    y = torch.ones_like(x, device=device) # this specific application of cuda is to create a tensor that utilizises the GPU (at least this is what I think it does. I could be wrong).
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) # I think that this line prints the tensor to the device specified. It says device conversion among uses.