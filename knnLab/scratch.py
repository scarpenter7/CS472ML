import operator as op
import numpy as np

coolDict = {'a':1000, 'b':3000, 'c': 100}
print(max(coolDict, key=coolDict.get))


list1 = [2, 5, 7, 6, 5, 4, 8, 4]
list2 = [2, 0, 7, 0, 0, 0, 0, 4]
print(sum([list1[i] == list2[i] for i in range(len(list2))]))

X = np.array([[ 0.3,  0.8], [-0.3,  1.8], [ 0.9,  0. ], [ 1., 1.]])
y = np.array([.5, .2])
# print(np.linalg.norm(X, y))

arr = np.array(['g', 'g', 'y', 'y', 'g', 'y'])
print(np.where(arr == 'y', 1, 0))