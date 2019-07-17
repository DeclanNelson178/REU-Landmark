import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
start_time = time.time()
torch.manual_seed(2)
import random
from mpl_toolkits import mplot3d
random.seed(2)

lm_epoch = 5000
set_size = 435
batch_size = 70
test_size = 135
num_lm = 20
size = (batch_size * 4) + num_lm
lbda = 1000
epoch = 5000
k_start = 3
k_lm  = 5
k_other = 5
m = 300
n = 16
categories = 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f = nn.Linear(16, 10, bias=True)
        self.f2 = nn.Linear(10, 2, bias=True)

    def encode(self, x):
        m = nn.LeakyReLU()
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


def original_clean():
    dataset = pd.read_csv('Parliment-1984.csv')
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    for i in range(0, 434):
        if y[i] == 'democrat':
            y[i] = 0
        elif y[i] == 'republican':
            y[i] = 1
    y = y.astype(int)

    for a in range(0, 434):
        for b in range(0, 16):
            if ('y' in X[a][b]):
                X[a][b] = 1
            elif ('n' in X[a][b]):
                X[a][b] = 0

    medians = []
    for x in range(0, 16):
        acceptable = []
        for z in range(0, 434):
            if ((X[z][x] == 1) or (X[z][x] == 0)):
                acceptable.append(X[z][x])
        med = np.median(acceptable)
        medians.append(int(med))

    for c in range(0, 434):
        for d in range(0, 16):
            if ((X[c][d] != 1) and (X[c][d] != 0)):
                X[c][d] = medians[d]
    X = X.astype(float)
    X = normalize(X)
    return X, y


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


def load_data():
    global batch_size, divisor
    data, labels = original_clean()
    test_data = data[300:, :]
    test_labels = labels[300:]
    data = data[:300, :]
    labels = labels[:300]

    N = NearestNeighbors(n_neighbors=k_start).fit(data).kneighbors_graph(data).todense()
    N = np.array(N)
    num_connections = N.sum(axis=0).argsort()[::-1]
    top_landmarks_idxs = num_connections[:num_lm]
    land_marks = data[top_landmarks_idxs, :]
    data = np.delete(data, top_landmarks_idxs, axis=0)
    landmark_neighbors = NearestNeighbors(n_neighbors=k_lm).fit(land_marks).kneighbors_graph(land_marks).todense()
    divisor = int(size / batch_size)
    batch_loader = np.zeros((divisor, batch_size + num_lm, n))
    batch_graph = np.zeros((divisor, batch_size + num_lm, batch_size + num_lm))
    for i in range(divisor):
        holder = data[batch_size * i: batch_size * (i + 1)]
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
    return batch_loader, land_marks, labels, data, batch_graph, top_landmarks_idxs, test_data, test_labels, landmark_neighbors

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
            # print('Epoch: %f, Step: %f, Loss: %.2f' % (num, batch_id + 1, loss.data.cpu().numpy()))


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
        # print('LM Epoch: %f, Loss: %.2f' % (num, loss.data.cpu().numpy()))


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


def evaluate(data, net, t, landmarks):
    out = net(torch.from_numpy(data).float(), False)
    print(time.time() - start_time)
    t = t.astype(float)
    out = out.detach().numpy()
    print('New score metric')
    print(score(out, t))
    cmap = colors.ListedColormap(['red','blue'])
    plt.scatter(out[:, 0], out[:, 1], c=t, cmap=cmap, marker='o')
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(out)
    vmeasure = v_measure_score(t, kmeans.labels_)
    print(vmeasure)
    # plt.show()


def run():
    global num_lm
    batch_loader, land_marks, labels, data, batch_graph, lmIndex, test_data, test_labels, landmark_neighbors = load_data()
    net = Net()
    opti = torch.optim.Adam(net.parameters(), weight_decay=1e-3)
    train_lms(lm_epoch, land_marks, net, opti, landmark_neighbors)
    train_net(epoch, batch_loader, net, opti, batch_graph)
    evaluate(test_data, net, test_labels, lmIndex)


def score(final_data, labels):
    m = np.size(final_data, 0)
    # Final Scoring
    eval_arr = np.zeros((10, 2))
    count_arr = np.zeros(10)
    for i in range(0, m):
        color_num = int(labels[i])
        eval_arr[color_num][0] += final_data[i][0]
        eval_arr[color_num][1] += final_data[i][1]
        count_arr[color_num] += 1
    for i in range(0, 10):
        eval_arr[i][0] = eval_arr[i][0] / count_arr[i]
        eval_arr[i][1] = eval_arr[i][1] / count_arr[i]
    count = 0
    for i in range(m):
        min_center = 0
        min_center_dist = (final_data[i][0] - eval_arr[0][0]) ** 2 + (final_data[i][1] - eval_arr[0][1]) ** 2
        for j in range(1, 10):
            if ((final_data[i][0] - eval_arr[j][0]) ** 2 + (final_data[i][1] - eval_arr[j][1]) ** 2) < min_center_dist:
                min_center_dist = (final_data[i][0] - eval_arr[j][0]) ** 2 + (final_data[i][1] - eval_arr[j][1]) ** 2
                min_center = j
        if min_center == labels[i]:
            count += 1
    return count / m

set_size = 435
batch_size = 70
test_size = 135
num_lm = 20
size = (batch_size * 4) + num_lm
lbda = 10000
lm_epoch = 5000
epoch = 5000
k_start = 3
k_lm  = 4
k_other = 4
m = 300
n = 16
categories = 2

run()