import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn.datasets
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import time

holder = pd.read_csv('Parliment-1984.csv', delimiter=',').values
labels = holder[:, 0]
data = holder[:, 1:]

print(labels)
size = len(labels)
attributes = len(data[0])

for i in range(len(labels)):
    if labels[i] == 'republican':
        labels[i] = 0
    else:
        labels[i] = 1


for i in range(size):
    for j in range(attributes):
        point = data[i][j]
        if point == 'y\\' or point == 'y':
            data[i][j] = 1
        elif point == 'n\\' or point == 'n' or point == 'n}':
            data[i][j] = 0

medians = []
for j in range(attributes):
    acceptable = []
    for i in range(size):
        if((data[i][j] == 1) or (data[i][j] == 0)):
            acceptable.append(data[i][j])
    med = np.median(acceptable)
    medians.append(int(med))

for i in range(size):
    for j in range(attributes):
        if data[i][j] != 1 and data[i][j] != 0:
            data[i][j] = medians[j]
F = open('pol_data.csv', 'w+')
for i in range(size):
    for j in range(attributes):
        F.write(str(data[i][j]) + ',')
    F.write('\n')




