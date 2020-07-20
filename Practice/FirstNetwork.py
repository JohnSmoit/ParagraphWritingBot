import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): #define neural network as Net

    def __init__(self): # constructor used to define layers n stuff
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3) #these are 2d convolutional layers I think
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120) # these are linear layers I think
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # this pushes data forward throught the network
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x): # determines features in the network like neurons and stuff
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

net.zero_grad()

print('conv1.bias.grad before backprop:') # update biases with backprop (normally put in a loop of sorts but this is an example)
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backprop:') # wala u got some updated stuff thanks to backprop
print(net.conv1.bias.grad)

