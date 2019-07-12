import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
start_time = time.time()
torch.manual_seed(2)
import random
import sys
random.seed(2)


# Hyper paramters
m, n, divisor = 0, 0, 0  # will reset these later

num_lm = 4
batch_size = 20
size = 300
linear_dim0 = 30
linear_dim1 = 100
linear_dim2 = 2
linear_dim3 = 10
linear_dim4 = 50
linear_dim5 = 2
lbda = 90000  # 100000, 90000
epoch = 500
squeeze = 2
set_random = False
temp_subset = num_lm + (batch_size * 5)
trainsize = 200

k_start = 3  # how you find landmarks based off of number of nearest neighbors
k_lm = 3  # number of landmarks each landmark has
k_other = 5  # number of landmarks each regular points has


def normalize(data):
    row = np.size(data, 0)
    col = np.size(data, 1)

    for j in range(col):
        col_sum = 0
        for i in range(row):
            col_sum = col_sum + data[i][j]
        col_sum = col_sum / row
        for i in range(row):
            data[i][j] = data[i][j] - col_sum
    return data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(linear_dim0, linear_dim1, bias=True)
        self.f2 = nn.Linear(linear_dim1, linear_dim2, bias=True)
        self.f3 = nn.Linear(linear_dim2, linear_dim3, bias=True)
        self.f4 = nn.Linear(linear_dim3, linear_dim4, bias=True)
        self.f5 = nn.Linear(linear_dim4, linear_dim5, bias=True)

    def encode(self, x):
        p = nn.LeakyReLU()
        x = p(self.f(x))
        x = self.f2(x)
        #x = p(self.f3(x))
        #x = p(self.f4(x))
        #x = self.f5(x)
        return x

    def decode(self, x):
        return x

    def forward(self, x, decode):
        x = self.encode(x)
        if decode:
            x = self.decode(x)
        return x


# load data set, select land marks at random and remove from data set, return a train_loader
def load_data(size, num_lm):
    global divisor, m, n, batch_size, set_random, trainsize, linear_dim0
    # load data from MNIST file
    # train_dataset = np.loadtxt("C:/TempAna/wdbc.data", delimiter=',', dtype = {'names':('ID','Diagnosis','Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity',
    #     'Concave Points','Symmetry','Fractal Dimension','Radius2','Texture2','Perimeter2','Area2','Smoothness2','Compactness2','Concavity2',
    #     'Concave Points2','Symmetry2','Fractal Dimension2','Radius3','Texture3','Perimeter3','Area3','Smoothness3','Compactness3','Concavity3',
    #     'Concave Points3','Symmetry3','Fractal Dimension3'), 'formats': (np.int,'|S1',np.float,np.float,np.float,np.float,np.float,np.float,np.float,
    #     np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,
    #    np.float,np.float,np.float,np.float,np.float,np.float,np.float)}) #Current set to the same for testing purposes
    #test_dataset = np.loadtxt("C:/TempAna/wdbc.data", delimiter=',', dtype = {'ID':('Diagnosis','Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity',
        #'Concave Points','Symmetry','Fractal Dimension')}) #Change later, perhaps?
    both_dataset = np.genfromtxt("home/tchaizhang/wdbc.data",dtype=np.float,delimiter=',',usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31),encoding=None)
    both_labels = np.genfromtxt("home/tchaizhang/wdbc.data",dtype=None,delimiter=',',usecols = (1),encoding=None)
    tempAll_labels = np.zeros(len(both_labels)) 
    for i in range(len(both_labels)):
        if both_labels[i]=='B':
            tempAll_labels[i] = 0
        else:
            tempAll_labels[i] = 1
    # todo remove this
    size = temp_subset
    m = size
    n = linear_dim0
    # todo if needed, normalize
    test_dataset = both_dataset[temp_subset:]
    train_dataset = both_dataset[:temp_subset]
    test_labels = torch.from_numpy(tempAll_labels[temp_subset:])
    train_labels = torch.from_numpy(tempAll_labels[:temp_subset])

    # normalize
    temp_data = normalize(train_dataset)
    temp_data = torch.from_numpy(temp_data)
    temp_labels = train_labels

    data = np.zeros((size, n))
    for i in range(size):
        data[i] = temp_data[i].view(-1, n)

    temp_data = data

    # make landmarks, select x random points in the data set
    land_marks = np.empty((num_lm, n))
    if set_random:
        for i in range(num_lm):
            index = random.randint(0, size - i)
            a = temp_data[index]
            land_marks[i] = temp_data[index]
            temp_data = np.delete(temp_data, index, axis=0)
            temp_labels = np.delete(temp_labels, index, axis=0)
    else:
        # N = NearestNeighbors(n_neighbors=k_start).fit(temp_data).kneighbors_graph(temp_data).todense()
        # N = np.array(N)
        # num_connections = N.sum(axis=0).argsort()[::-1]
        # top_landmarks_idxs = num_connections[:num_lm]
        # land_marks = temp_data[top_landmarks_idxs, :]
        # temp_data = np.delete(temp_data, top_landmarks_idxs, axis=0)
        # try and choose a single landmark from every number label
        top_landmarks_idxs = np.zeros(num_lm, dtype=np.int32)
        # used_nums = np.zeros(num_lm, dtype=np.int8)
        num_each = int(num_lm / 10)
        # for i in range(m):
        #     index = temp_labels[i].numpy()  # index is the label number
        #     i = int(i)
        #     if used_nums[index] == 0:
        #         top_landmarks_idxs[index] = i
        #         used_nums[index] = 1
        # land_marks = temp_data[top_landmarks_idxs, :]
        # temp_data = np.delete(temp_data, top_landmarks_idxs, axis=0)

        for i in range(10):
            count = 0
            for j in range(m):
                if temp_labels[j] == i and count < num_each:
                    index = i * num_each + count
                    count += 1
                    top_landmarks_idxs[index] = j
        land_marks = temp_data[top_landmarks_idxs, :]
        temp_data = np.delete(temp_data, top_landmarks_idxs, axis=0)

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
    return batch_loader, temp_data, batch_graph, landmark_neighbors, test_dataset, land_marks, test_labels, train_labels


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
            loss = (1 / lbda) * nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())  # lmbda*(1/output.var(dim=0)[0] + 1/output.var(dim=0)[1]) #lmbd
            opti.zero_grad()
            loss.backward()
            opti.step()
            #print('Epoch: %f, Step: %f, Loss: %.2f' % (num, batch_id + 1, loss.data.cpu().numpy()))


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
        loss = (1 / lbda) * nbr_distance + lbda * (1 / out[:, 0].var() + 1 / out[:, 1].var())  # lmbda*(1/output.var(dim=0)[0] + 1/output.var(dim=0)[1]) #lmbd
        opti.zero_grad()
        loss.backward()
        opti.step()
        #print('LM Epoch: %f, Loss: %.2f' % (num, loss.data.cpu().numpy()))


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


def evaluate(test_loader, net, num_points, labels):
    global linear_dim0
    n = linear_dim0
    holder_graph = np.empty((num_points, 2))
    holder_labels = np.empty((num_points))
    correct = 0
    total = 0
    x = 0
    final_score = 0
    for images in test_loader:
        x += 1
        images = images.reshape(-1, n)
        out = net(images.float(), False)
        none, predicted = torch.max(out.data, 1)
        for i in range(len(predicted)):
            if predicted[i] == 1:
                predicted[i] = 0
            else:
                predicted[i] = 1
        total += len(labels)
        correct += (predicted == labels.long()).sum().item()
        #print('Hyperparameters: num_lm: %f, batch_size: %f, lbda: %f, k_start: %f, k_lm: %f, k_other: %f, First Dimension: %f, Second Dimension: %f' % (num_lm,batch_size,lbda,k_start,k_lm,k_other, linear_dim1, linear_dim2))
        #print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        print('%f,%f,%f,%f,%f,%f,%f,%f,%f' % ((100 * correct / total), num_lm, batch_size , lbda, k_start , k_lm, k_other , linear_dim1 , linear_dim2 ))
        out = out.detach().numpy()
        holder_graph = out
        holder_labels = labels
        x += 1
        final_score = 100 * correct / total
    return final_score


def run():
    num_points = 10000
    global num_lm
    data_loader, data, batch_graph, landmark_neighbors, test_dataset, land_marks, test_labels, train_labels = load_data(size, num_lm)
    net = Net()
    net = net.float()
    opti = torch.optim.SGD(net.parameters(), lr=.0001, momentum=.1)
    for i in net.modules():
        if isinstance(i, nn.Linear):
            i.weight.data.normal_(0, .1)

    train_lms(epoch, land_marks, net, opti, landmark_neighbors)
    train_net(epoch, data_loader, net, opti, batch_graph)
    # change test_loader batch size to increase number of points tested on
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=num_points, shuffle=False)
    return evaluate(test_loader, net, num_points, test_labels)

number = int(sys.argv[1])

num_lm = 30
batch_size = 60
size = 560
linear_dim0 = 30
linear_dim1 = 100
linear_dim2 = 8
temp_subset = num_lm + (batch_size * 5)

#linear_dim2 = 2 + number
#linear_dim1 = 700
lbda = 10000 + (10000 * number)
k_lm = 1
k_other = 1
k_start = 1
run()
while k_start < 20:
    k_lm = 1
    batch_size = 60
    while k_lm < 20:
        k_other = 1
        batch_size = 60
        while k_other < 20:
            try:
                batch_size = 60
                temp_subset = num_lm + (batch_size * 5)
                run()
            except:
                print("Error with'Hyperparameters: num_lm: %f, batch_size: %f, lbda: %f, k_start: %f, k_lm: %f, k_other: %f, First Dimension: %f, Second Dimension: %f" % (num_lm,batch_size,lbda,k_start,k_lm,k_other, linear_dim1, linear_dim2))
            k_other += 1 
        k_lm += 1
    k_start += 1


run()