import torch
from torch.autograd import Variable
import numpy as np

#Part 1: Start of PyTorch
#Initialize a matrix
x1 = torch.Tensor(5, 3)
print(x1)

x2 = torch.rand(5, 3)
print(x2)

#Different ways of addition
y1 = x1 + x2
y2 = torch.add(x1, x2)

#Resizing
x = torch.rand(5, 3)
y = x.view(15)
z = x.view(-1, 3)

#Convert a torch tensor to numpy array
a = torch.ones(5)
b = a.numpy()
print(b)

#Convert a numpy array to a torch tensor
a = np.ones(5)
b = torch.from_numpy(a)

#Part 2: Autograd: automatic differentiation
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)
y = x*x
print('The gradient of y on x: ', y.grad_fn)


z = y * y * 3
out = z.mean()

print(z, out)


