import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):

    # Слои, в которых нет параметров для обучения можно не создавать, а брать из переменной F
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(1))

        self.net.add_module('L1', nn.Linear(3*28*28,588))       
        self.net.add_module('sf_1', nn.Sigmoid())
        
        self.net.add_module('L2', nn.Linear(588,244))       
        self.net.add_module('relu_2', nn.ReLU())


        self.net.add_module('L3', nn.Linear(244,122))       
        self.net.add_module('relu_3', nn.ReLU())


        self.net.add_module('L4', nn.Linear(122,61))       
        self.net.add_module('relu_4', nn.ReLU())


        self.net.add_module('L5', nn.Linear(61,10))       
        self.net.add_module('sf_5', nn.Softmax())

    def forward(self, x):
        x = self.net(x)
        return x