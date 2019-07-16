import cvxpy as cp
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
start_time = time.time()
torch.manual_seed(2)
import random
import sys
random.seed(2)
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import colors

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

def load_data_whole():
    both_dataset = np.genfromtxt("wdbc.data",dtype=np.float,delimiter=',',usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31),encoding=None)
    both_labels = np.genfromtxt("wdbc.data",dtype=None,delimiter=',',usecols = (1),encoding=None)
    tempAll_labels = np.zeros(len(both_labels)) 
    for i in range(len(both_labels)):
        if both_labels[i]=='B':
            tempAll_labels[i] = 0
        else:
            tempAll_labels[i] = 1

    both_labels = tempAll_labels
    
    # make landmarks, select x random points in the data set
    return both_dataset, both_labels


def pca(r):
    #normalize
    #PCA
    start_time = time.time()
    data, labels = load_data_whole()

    pca = PCA(n_components=r)
    data = pca.fit_transform(data)

    return data, labels

def mvu(k, r, file):
    #x = np.loadtxt(file, delimiter=" ")
    x = file
    #labels = x[:,0]
    #x = np.delete(x, 0, 1)

    # m is the number of data points
    m = np.size(x, 0)

    temp = []
    for i in range(0, m):
        temp.append(x[i])
    x = np.asarray(temp)

    n = len(x[0])  # number of parameters

    # normalize the x matrix
    for i in range(0, n):
        col_sum = 0
        for j in range(0, m):
            col_sum += x[j][i]
        mean = col_sum / m
        for j in range(0, m):
            x[j][i] = x[j][i] - mean
    print('Matrix normalized')
    # find the square distance matrix
    sd_matrix = np.zeros((m, m))
    for i in range(0, m):
        for j in range(i + 1, m):
            dist_sum = 0
            for l in range(0, n):
                dist_sum += (x[i][l] - x[j][l]) ** 2
            sd_matrix[i][j] = dist_sum
            sd_matrix[j][i] = dist_sum

    # define the neighborhood
    neighborhood = np.zeros((m, m))
    for i in range(0, m):
        new_row = list()
        for j in range(0, m):
            new_row.append((j, sd_matrix[i][j]))
        row = sorted(new_row, key=lambda tup: tup[1])
        for j in range(0, k):
            neighborhood[row[j][0]][i] = 1
    neighborhood = neighborhood.transpose()


    # code constraints and solve problem
    kernel = cp.Variable((m, m), symmetric=True)
    constraints = [kernel >> 0]
    constraints += [cp.sum(kernel) == 0]
    for i in range(0, m):
        for j in range(0, m):
            if neighborhood[i][j] == 1:
                constraints += [kernel[i][i] + kernel[j][j] - (2*kernel[i][j]) == sd_matrix[i][j]]
    prob = cp.Problem(cp.Maximize(cp.trace(kernel)), constraints)
    prob.solve()
    kernel = kernel.value

    # find all eigenvalues and eigenvectors
    e_vals, e_vecs = np.linalg.eig(kernel)

    # pick the top r to create the principal component matrix
    for i in range(0, m - r):
        e_vecs = np.delete(e_vecs, -1, 1)
        e_vals = np.delete(e_vals, -1)

    # take the singular square roots of the kernel's engenvalues and apply along diagonal
    lbda = np.diag(e_vals ** 0.5)
    # multiply the pc matrix by the x matrix to get the final answer
    final_data = lbda.dot(e_vecs.T).T
    return final_data

def score(final_data, labels):
    m = np.size(final_data, 0)
    #Final Scoring
    eval_arr = np.zeros((10,2))
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
            if ((final_data[i][0] - eval_arr[j][0]) ** 2 + (final_data[i][1] - eval_arr[j][1]) ** 2) < min_center_dist :
                min_center_dist = (final_data[i][0] - eval_arr[j][0]) ** 2 + (final_data[i][1] - eval_arr[j][1]) ** 2
                min_center = j
        if min_center == labels[i]:
            count += 1
    return count/m


def graph(final_data, labels):
    m = np.size(final_data, 0)
    cmap = colors.ListedColormap(['blue','red'])
    plt.scatter(final_data[:,0], final_data[:,1], c=labels[0:m], cmap=cmap, marker='.', s=100, alpha=0.45)
    plt.show()

def run():
    pca_data_set, num_labels = pca(linear_dim1)
    data_set = mvu(k, linear_dim2, pca_data_set)
    print("MVU complete")
    accuracy_score = score(data_set, num_labels)
    print(accuracy_score)
    if accuracy_score >= 0:
        print(time.time() - start_time)
        graph(data_set, num_labels)


# number = int(sys.argv[1])

# num_lm = 30
# batch_size = 60
# size = 560
# linear_dim0 = 30
# linear_dim1 = 100
# linear_dim2 = 8
# epoch = 500
# k = 2
# lbda = 10000

# #linear_dim2 = 2 + number
# #linear_dim1 = 700
# lbda = 10000 + (number*10000)

# linear_dim1 = 10
# linear_dim2 = 2
# k = 2
# while linear_dim1 < 100:
# 	while linear_dim2 < 15:
# 		while k < 10:
# 			try:
# 				run()
# 			except:
# 				print(Error)
# 		k += 1
# 	linear_dim2 += 2
# linear_dim1 += 10

linear_dim1 = 15
linear_dim2 = 2
linear_dim0 = 30
k = 2

run()