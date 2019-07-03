import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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
from mpl_toolkits import mplot3d
random.seed(2)


# Hyper paramters
m, n, divisor = 0, 0, 0  # will reset these later

num_lm = 20
batch_size = 200
size = 10000
linear_dim1 = 500
linear_dim2 = 250
linear_dim3 = 125
linear_dim4 = 50
linear_dim5 = 2
lbda = 100000  # 1000000
epoch = 5000
squeeze = 2
set_random = True
temp_subset = num_lm + (batch_size * 10)

k_start = 3  # how you find landmarks based off of number of nearest neighbors
k_lm = 4  # number of landmarks each landmark has
k_other = 3  # number of landmarks each regular points has


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(28 * 28, linear_dim1, bias=True)
        self.f2 = nn.Linear(linear_dim1, linear_dim2, bias=True)
        self.f3 = nn.Linear(linear_dim2, linear_dim3, bias=True)
        self.f4 = nn.Linear(linear_dim3, linear_dim4, bias=True)
        self.f5 = nn.Linear(linear_dim4, linear_dim5, bias=True)

    def encode(self, x):
        p = nn.LeakyReLU()
        x = p(self.f(x))
        x = p(self.f2(x))
        x = p(self.f3(x))
        x = p(self.f4(x))
        x = self.f5(x)
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
    global divisor, m, n, batch_size, set_random
    print("starting to load")
    # load data from MNIST file
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                              download=True)
    print("done loading")
    m = size
    n = 28 ** 2
    # todo if needed, normalize
    print("making temp")
    temp_data = train_dataset.data
    temp_labels = train_dataset.train_labels

    # todo remove this
    temp_labels = temp_labels[:temp_subset]
    size = temp_subset
    m = temp_subset

    data = np.zeros((size, 28 ** 2))
    for i in range(size):
        data[i] = temp_data[i].view(-1, 28 ** 2)

    temp_data = data
    # make landmarks, select x random points in the data set
    land_marks = np.empty((num_lm, n))
    top_landmarks_idxs = []
    print("picking landmarks")
    if set_random:
        for i in range(num_lm):
            index = random.randint(0, size - i)
            a = temp_data[index]
            land_marks[i] = temp_data[index]
            temp_data = np.delete(temp_data, index, axis=0)
            temp_labels = np.delete(temp_labels, index, axis=0)
    else:
        N = NearestNeighbors(n_neighbors=k_start).fit(temp_data).kneighbors_graph(temp_data).todense()
        N = np.array(N)
        num_connections = N.sum(axis=0).argsort()[::-1]
        top_landmarks_idxs = num_connections[:num_lm]
        land_marks = temp_data[top_landmarks_idxs, :]
        temp_data = np.delete(temp_data, top_landmarks_idxs, axis=0)

    print("neighborhood creation")
    landmark_neighbors = NearestNeighbors(n_neighbors=k_lm).fit(land_marks).kneighbors_graph(land_marks).todense()
    divisor = int(size / batch_size)
    batch_loader = np.zeros((divisor, batch_size + num_lm, n))
    batch_graph = np.zeros((divisor, batch_size + num_lm, batch_size + num_lm))
    for i in range(divisor):
        holder = temp_data[batch_size * i: batch_size * (i + 1)]
        holder_graph = NearestNeighbors(n_neighbors=k_other).fit(land_marks).kneighbors_graph(holder).todense()
        for j in range(batch_size):  # copy over the holder graph
            for l in range(num_lm):
                if holder_graph[j, l] == 1:
                    batch_graph[i, j, l + batch_size] = 1
                    batch_graph[i, l + batch_size, j] = 1
        for j in range(num_lm):  # copy over landmark neighbors
            for l in range(j, num_lm):
                if landmark_neighbors[j, l] == 1 and j != l:
                    batch_graph[i, j + batch_size, l + batch_size] = 1
                    batch_graph[i, l + batch_size, j + batch_size] = 1
        holder = np.concatenate((holder, land_marks))
        batch_loader[i] = holder
    batch_size += num_lm
    return batch_loader, temp_data, batch_graph, landmark_neighbors, test_dataset, land_marks


def train_net(epoch, data, net, opti, batch_graph):
    global divisor, batch_size
    for num in range(epoch):
        for batch_id in range(divisor):
            batch = torch.from_numpy(data[batch_id]).float()
            batch = batch.view(batch_size, -1)
            batch_distances = pairwise_distances(batch)
            nbr_graph_tensor = torch.from_numpy(batch_graph[batch_id]).float()
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
            print('Epoch: %f, Step: %f, Loss: %.2f' % (num, batch_id + 1, loss.data.cpu().numpy()))


def train_lms(epoch, land_marks, net, opti, landmark_neighbors):
    for num in range(epoch):
        global lbda
        batch = torch.from_numpy(land_marks).float().view(num_lm, -1)
        batch_distances = pairwise_distances(batch)
        neighbor_graph = torch.from_numpy(landmark_neighbors).float()
        batch_distances_masked = batch_distances * neighbor_graph.float()
        out = net(batch, False)
        output_distances = pairwise_distances(out)
        output_distances_masked = output_distances * neighbor_graph.float()
        nbr_diff = torch.abs((output_distances_masked - batch_distances_masked))
        nbr_distance = nbr_diff.norm()
        loss = nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())  # lmbda*(1/output.var(dim=0)[0] + 1/output.var(dim=0)[1]) #lmbd
        opti.zero_grad()
        loss.backward()
        opti.step()
        print('LM Epoch: %f, Loss: %.2f' % (num, loss.data.cpu().numpy()))


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


def evaluate(data, net):
    temp_data = data.data
    temp_labels = data.test_labels.numpy()
    data = torch.zeros((size, 28 ** 2))
    for i in range(size):
        data[i] = temp_data[i].view(-1, 28 ** 2)
    temp_data = data

    # data = np.concatenate((data, landmarks), axis=0)
    out = net(temp_data, False)
    # print(time.time() - start_time)
    out = out.detach().numpy()
    plt.scatter(out[:, 0], out[:, 1], c=temp_labels, marker='o')
    plt.show()


def run():
    global num_lm
    data_loader, data, batch_graph, landmark_neighbors, test_dataset, land_marks = load_data(size, num_lm)
    print("done loading data")
    net = Net()
    opti = torch.optim.Adam(net.parameters(), weight_decay=1e-3)
    print("starting training")
    train_lms(epoch, land_marks, net, opti, landmark_neighbors)
    train_net(epoch, data_loader, net, opti, batch_graph)
    evaluate(test_dataset, net)


run()