"""
@author Zach Stoebner
@date October 2019
@details Basic torchvision cnn --> derived from PyTorch 60-min blitz [https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py]
"""

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet,self).__init__()
        model = []
        model += nn.Conv2d(3, 6, 5)
        model += nn.ReLU()
        model += nn.MaxPool2d(2, 2)
        model += nn.Conv2d(6, 16, 5)
        model += nn.ReLU()
        model += nn.MaxPool2d(2, 2)
        model += nn.Flatten()
        model += nn.Linear(16 * 5 * 5, 120)
        model += nn.ReLU()
        model += nn.Linear(120, 84)
        model += nn.ReLU()
        model += nn.Linear(84, 10)

        # unwrap model into a sequential container
        self.model = nn.Sequential(*model)

    # forward prop
    def forward(self, x):
        out = self.model(x)
        return out

def train(net=BasicNet()):

    # loading dataset
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print('Finished Training')

###NOTE need to add test()
