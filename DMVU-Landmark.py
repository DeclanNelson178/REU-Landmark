import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()
torch.manual_seed(2)
import random


# Hyper paramters
size = 0
batch_size = 0
num_lm = 0
m = 0 # number of data points
n = 0 # number of dimensions

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, linear_dim1)
        self.fc2 = nn.Linear(linear_dim1, linear_dim2)
        # self.fc_final = nn.Linear(linear_dim2, 10)

    def forward(self, x):
        s = nn.Softmax()
        # x = F.sigmoid(self.fc1(x))
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # x = s(x)

        return x


# load data set, select land marks at random and remove from data set, return a train_loader
def load_data(size, batch_size, num_lm):
    # import data
    data, labels = sklearn.datasets.make_swiss_roll(size)
    # data = data.normalize()
    m = np.size(data, 0)
    n = np.size(data, 1)

    # make landmarks, select x random points in the data set
    land_marks = np.empty((num_lm, n))
    for i in range(num_lm):
        index = random.randint(0, m)
        land_marks[i] = data[index]
    divisor = int(size / batch_size)
    start = 0
    batch_loader = np.zeros((divisor, batch_size + num_lm, n))
    for i in range(divisor):
        holder = data[start: start + batch_size]
        holder = np.concatenate((holder, land_marks))
        batch_loader[i] = holder
        start += batch_size

    batch_size += num_lm
    # make loader
    # return loader, landmarks
    return batch_loader, land_marks


# def normalize(data)
#
# def train_net(net, ):
#
#
# def run():
#     data_loader, land_marks = load_data(size, batch_size)
#
#     net = Net()
#
#     train_net(net, land_marks)

loader, land_marks = load_data(250, 50, 9)
print(loader)
