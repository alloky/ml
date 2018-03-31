import os
import torch
import torch.nn as nn
import pickle
from torch.nn.functional import softmax
from torch.autograd import Variable
from mnist_model import Net
from optparse import OptionParser
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from tqdm import *

import random
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
   

# разбираем аргументы коммандной строки
parser = OptionParser("Train cifar10 neural network")

parser.add_option("-i", "--input", dest="input", default='./',
                  help="Cifar data root directory")  # рутовый каталог откуда беруться данные

parser.add_option('-m',"--model", dest="model",
                  help="Model base path ") # путь к файлу модели

parser.add_option("-o", "--out", dest="out", type='string', default='solution.csv',
                  help="Path to tensorboard log")   # куда складывать результаты

def eval(options):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # Создаем модель, нужно сделать иплементацию
    print("Creating model...")
    net = Net().cuda()
    net.eval()
    # Критерий кросс энтропия проверим, что у нас вск сходится
    criterion = nn.CrossEntropyLoss().cuda()

    # загружаем сеть
    cp_dic = torch.load(options.model)
    net.load_state_dict(cp_dic)

    transform_test = transforms.Compose([
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) #нормализация данных
    ])

    # данные для теста

    testset = mnist.MNIST(options.input, train=False, transform=transforms.ToTensor())
    
    p_test = pickle.load(open("mnist_test.pkl","rb"))['data']


    print(torch.FloatTensor(p_test[0]).view(28,28).unsqueeze())
    print(transform_test(p_test[0]))
    # for i in range(len(p_test)):
    #     p_test[i][0] = transform_test(p_test[i][0])
    return
    testloader = DataLoader(p_test, batch_size=16,
                                             shuffle=False, num_workers=2)

    test_loss = 0
    print('Test model: ')

    ofile = open(options.out, 'w')
    print("id,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9", file=ofile)

    flag = True
    for bid, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
        inputs, labels = data

        # получаем переменные Variable
        inputs, labels = Variable(inputs, volatile=True).cuda(), Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        if(flag):
            print(inputs[0][0])
            flag = False
        # считаем ошибку
        loss = criterion(outputs, labels)
        test_loss += loss.data[0]
        # считаем какие классы мы предсказали и сохраняем для
        # последующего расчета accuracy
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.data).squeeze()
        for i in range(outputs.size(0)):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

        # печатаем для каждого класса вероятности
        probs = softmax(outputs)
        for sid, sample in enumerate(probs.data):
            s = '%d' % ((bid * 16)+sid)
            for prob in sample:
                s += ',%f'%prob
            print(s, file=ofile)

    test_loss /= len(testloader)
    # расчитываем accuracy
    accuracy = {}
    avg_accuracy = 0
    for i in range(10):
        accuracy[classes[i]] = 100 * class_correct[i] / class_total[i]
        avg_accuracy += accuracy[classes[i]]

    print("Final cross entropy loss: %0.5f"%test_loss, "Final accuracy: %0.3f"%(avg_accuracy/10) )

if __name__ == '__main__':
    (options, args) = parser.parse_args()
    if options.model is None or not os.path.exists( options.model ):
        print ('Model file does not exist or empty', options.model)
        exit(1)
    eval(options)