from scipy.spatial import distance_matrix
import numpy as np
import random
from matplotlib import pyplot as plt

data = np.array([[.8, .7],
                 [-.1, .2],
                 [.9, .8],
                 [0, .2],
                 [.2, .1]])

#print(data)
#print(np.vstack((data[0], data[1], data[2])))
dMatrix = distance_matrix(data, data, p=1)
#print(dMatrix)
#print(type(dMatrix))

data2 = np.array([[.8, .7, .4],
                 [-.1, .2, 0],
                 [.9, .8, .5],
                 [0, .2, -.4],
                 [.2, .1, -1.]])

#print(data2)
dMatrix2 = distance_matrix(data2, data2, p=1)
#print(dMatrix2)
np.fill_diagonal(dMatrix2, np.inf)
#print(dMatrix2)
indices = np.argwhere(dMatrix2 == np.min(dMatrix2))[0]
#print(indices)
#print(np.max(indices))
#print(np.min(indices))

row1 = dMatrix2[indices[0], :]
row1[row1 == np.inf] = 0
row2 = dMatrix2[indices[1], :]
row1[row2 == np.inf] = 0
#print(row1)
#print(row2)
stackRows = np.vstack((row1, row2))
newRow = np.amin(stackRows, axis=0)
newRow = newRow[newRow != 0]
#print(newRow)
newRow0 = np.concatenate((newRow, np.array([np.inf])))
newRowT = newRow[:, np.newaxis]
#print(newRowT)

dMatrix2 = np.delete(dMatrix2, np.max(indices), 0)
dMatrix2 = np.delete(dMatrix2, np.max(indices), 1)
dMatrix2 = np.delete(dMatrix2, np.min(indices), 0)
dMatrix2 = np.delete(dMatrix2, np.min(indices), 1)
#print(dMatrix2)
dMatrix2 = np.hstack((dMatrix2, newRowT))
#print(dMatrix2)
dMatrix2 = np.vstack((dMatrix2, newRow0))
#print(dMatrix2)

def test():
    x = np.array([1,2,10,14,4,2])
    return np.argmax(x)
#print(test())


data = np.array([[.8, .7],
                 [-.1, .2],
                 [.9, .8],
                 [0, .2],
                 [.2, .1]])
print(np.ndim(data))
y = np.array(['a', 'b', 'c', 'd', 'e'])
print({tuple(data[i]):y[i] for i in range(len(y))})
print(np.ndim(np.array([1,2,3])))
indices = random.sample(range(100), 50)
print(len(set(indices)))
print(indices)
dopeList = [[]]*5
print(dopeList)

