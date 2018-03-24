import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):

    # Слои, в которых нет параметров для обучения можно не создавать, а брать из переменной F
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(3))
        self.net.add_module('conv_1', nn.Conv2d(3, 6, 5))
        self.net.add_module('bn_1', nn.BatchNorm2d(6))
        self.net.add_module('do_1', nn.Dropout(0.5))
        self.net.add_module('relu_1', nn.ReLU())
        self.net.add_module('pool_1', nn.MaxPool2d(2, 2))
        self.net.add_module('conv_2', nn.Conv2d(6, 16, 5))
        self.net.add_module('relu_2', nn.ReLU())
        self.net.add_module('pool_2', nn.MaxPool2d(2, 2))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x