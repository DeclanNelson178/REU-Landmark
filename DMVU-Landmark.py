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
m, n, divisor = 0, 0, 0  # will reset these later

size = 1000
batch_size = 250
num_lm = 20
linear_dim1 = 10
linear_dim2 = 2
lbda = 100000
epoch = 5000
squeeze = 2
set_random = False
k = 3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(3, linear_dim1, bias=True)
        self.f2 = nn.Linear(linear_dim1, linear_dim2, bias=True)

    def encode(self, x):
        m = nn.PReLU()
        x = m(self.f(x))
        x = self.f2(x)
        return x

    def decode(self, x):
        return x

    def forward(self, x, decode):
        x = self.encode(x)
        if decode:
            x = self.decode(x)
        return x


def normalize(data):
    global squeeze, m, n
    for j in range(n):
        col_sum = 0
        for i in range(m):
            col_sum += data[i][j]
        col_sum /= m
        for i in range(m):
            data[i][j] -= col_sum
    initGraph = data.transpose()
    initGraph[1] = initGraph[1] / squeeze
    data = initGraph.transpose()
    return data


# load data set, select land marks at random and remove from data set, return a train_loader
def load_data(size, num_lm):
    global divisor, m, n, batch_size
    # import data
    data, labels = sklearn.datasets.make_swiss_roll(size)
    m = np.size(data, 0)
    n = np.size(data, 1)
    data = normalize(data)

    # make landmarks, select x random points in the data set
    land_marks = np.empty((num_lm, n))
    used_index = np.zeros(m)
    if random:
        for i in range(num_lm):
            index = random.randint(0, m)
            if used_index[index] == 0:
                land_marks[i] = data[index]
                used_index[index] = 1
            else:
                i -= 1
    else:
        N = NearestNeighbors(n_neighbors=k).fit(data).kneighbors_graph(data).todense()
        N = np.array(N)
        num_connections = N.sum(axis=0).argsort()[::-1]
        top_landmarks_idxs = num_connections[:num_lm]
        land_marks = data[top_landmarks_idxs, :]
    divisor = int(size / batch_size)
    batch_loader = np.zeros((divisor, batch_size + num_lm, n))
    for i in range(divisor):
        holder = data[batch_size * i: batch_size * (i + 1)]
        holder = np.concatenate((holder, land_marks))
        batch_loader[i] = holder
    batch_size += num_lm
    return batch_loader, land_marks, labels, data


# def normalize(data)
#
def train_net(epoch, data, net, opti, nbr_graph_tensor):
    global divisor, batch_size
    for num in range(epoch):
        for batch_id in range(divisor):  # todo switch batch and epoch
            batch = torch.from_numpy(data[batch_id]).float()
            batch = batch.view(batch_size, -1)
            batch_distances = pairwise_distances(batch)
            batch_distances_masked = batch_distances * nbr_graph_tensor.float()
            global lbda
            out = net(batch, False)
            output_distances = pairwise_distances(out)
            # Multiply the distances between each pair of points with the neighbor mask
            output_distances_masked = output_distances * nbr_graph_tensor.float()
            # Find the difference between |img_i - img_j|^2 and |output_i - output_j|^2
            nbr_diff = torch.abs((output_distances_masked - batch_distances_masked))
            nbr_distance = nbr_diff.norm()
            loss = nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())  # lmbda*(1/output.var(dim=0)[0] + 1/output.var(dim=0)[1]) #lmbd
            opti.zero_grad()
            loss.backward()
            opti.step()
            print('Epoch: %f, Step: %f, Loss: %.2f' % (epoch, batch_id + 1, loss.data.cpu().numpy()))


def make_neighborhood(batch):
    global n, batch_size
    neighborhood = torch.zeros(batch_size, batch_size, dtype=torch.float)  # set the type
    for i in range(batch_size - num_lm, batch_size):
        for j in range(0, batch_size):
            neighborhood[i][j] = 1.0
            neighborhood[j][i] = 1.0
    return neighborhood


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1) # square every element, sum, resize to list
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def evaluate(data, net, t):
    out = net(torch.from_numpy(data).float(), False)
    # print(time.time() - start_time)
    out = out.detach().numpy()
    plt.scatter(out[:, 0], out[:, 1], c=t, marker='o')
    plt.show()


def run():
    global num_lm
    data_loader, land_marks, labels, data = load_data(size, num_lm)
    net = Net()
    opti = torch.optim.Adam(net.parameters(), weight_decay=1e-3)
    neighborhood = make_neighborhood(data_loader[0])
    train_net(epoch, data_loader, net, opti, neighborhood)
    evaluate(data, net, labels)


run()

