import numpy as np
import statistics as st

targets = np.array([[0,0,1,0],
                    [0,0,1,0],
                    [1,0,0,0],
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,0,1]])
y = np.array([2, 3, 7, 5, 1])
print(np.sum(targets, axis=0))
print(targets[[1, 2, 4], :])
print(y[[1, 3, 4]])
coolList = np.array([1, 3, 3, 1, 1, 4, 5])
coolVals = np.array([0, 2, 9, 1, 3, 4, 5])
print(coolList)
print(st.mode(coolList))

coolMap = {coolList[i]: coolVals[i] for i in range(len(coolList))}
print(coolMap)
print(min(coolMap.keys()))
print(coolMap.get(min(coolMap.keys())))