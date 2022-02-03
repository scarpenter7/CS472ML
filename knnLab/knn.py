from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
import statistics as st


class KNNClassifier(BaseEstimator ,ClassifierMixin):
    def __init__(self, outputClasses, columntype=[], weight_type='inverse_distance', regressionVals = False, kList = [3]): # add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] = True or if nominal[categoritcal] = False.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype  # Note This won't be needed until part 5
        self.weight_type = weight_type
        self.kList = kList
        self.outputClasses = outputClasses
        self.regressionVals = regressionVals

    def fit(self, X, y):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.X = X
        self.y = y
        return self

    def predict(self, data):
        """ Predict all classes for a dataset X
        Args:
            data (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        # print("data: " + str(data))
        dists = np.linalg.norm(self.X-data, axis=1)
        # dists = np.linalg.norm(self.X - X)
        if self.weight_type == "inverse_distance":
            if self.regressionVals:
                return [self.invDistReg(dists, k) for k in self.kList]
            return [self.invDist(dists, k) for k in self.kList]
        if self.regressionVals:
            return [self.majorityReg(dists, k) for k in self.kList]
        return [self.majority(dists, k) for k in self.kList]

    def invDist(self, dists, k):
        # Find k nearest neighbors
        # print("k = " + str(k))
        sortedIndexes = np.argpartition(dists, k)
        nearestIndices = sortedIndexes[:k]

        # Compute inverse distances and add them up for each output class
        distWeightDict = {outputClass: 0 for outputClass in self.outputClasses}
        for i in nearestIndices:
            dist = dists[i]
            invDist = 1 / (dist**2)
            outputClass = self.y[i]
            distWeightDict[outputClass] += invDist

        # Return the output class of the highest sum of inverse distances
        nearest = max(distWeightDict, key=distWeightDict.get)
        return nearest

    def invDistReg(self, dists, k):
        # Find k nearest neighbors
        sortedIndexes = np.argpartition(dists, k)
        nearestIndices = sortedIndexes[:k]

        # Compute inverse distances and add them up for each output class
        weights = []
        vals = []
        for i in nearestIndices:
            dist = dists[i]
            invDist = 1 / (dist ** 2)
            weights.append(invDist)
            vals.append(invDist * self.y[i])

        # Return the output class of the highest sum of inverse distances
        regVal = sum([vals[i] for i in range(len(vals))]) / sum([weights[i] for i in range(len(weights))])
        return regVal

    def majority(self, dists, k):
        # Find k nearest neighbors and return majority
        sortedIndexes = np.argpartition(dists, k)
        votes = self.y[sortedIndexes[:k]]
        majority = st.mode(votes)
        return majority

    def majorityReg(self, dists, k):
        # Find k nearest neighbors and return majority
        sortedIndexes = np.argpartition(dists, k)
        votes = self.y[sortedIndexes[:k]]
        majority = sum(votes) / k
        return majority


    """def computeDist(self, pt1, pt2):
        return np.sum(np.abs((pt1 - pt2)**2))**(1/2)"""

    # Returns the Mean score given input data and labels
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        if self.regressionVals:
            # return MSE instead
            return [sum([abs(self.predict(pt)[k] - y[i])**2 for i, pt in enumerate(X)]) / len(X) for k in range(len(self.kList))]
        return [sum([self.predict(pt)[k] == y[i] for i, pt in enumerate(X)]) / len(X) for k in range(len(self.kList))]



# Tests
def parse(filename):
    file = open(filename)
    lines = file.readlines()
    y = []
    X = []
    for line in lines:
        nums = [float(num) for num in line.split(',')]
        input = nums[:-1]
        output = nums[-1]
        X.append(input)
        y.append(output)
    X = np.array(X)
    y = np.array(y)
    return X, y

def test1():
    print("START TEST 1:")
    X, y = parse("hwData.txt")
    outputClasses = [0, 1]
    colTypes = [True, True, False]
    hw = KNNClassifier(outputClasses, columntype=colTypes, weight_type="no_weight").fit(X, y)
    prediction = hw.predict(np.array([.5, .2]))
    print(prediction)
    print("END TEST 1")
    print()

def test2():
    print("START TEST 2:")
    X, y = parse("hwData.txt")
    outputClasses = [0, 1]
    colTypes = [True, True, False]
    hw = KNNClassifier(outputClasses, columntype=colTypes).fit(X, y)
    prediction = hw.predict(np.array([.5, .2]))
    print(prediction)
    print("END TEST 2")
    print()

def test3():
    print("START TEST 3:")
    X, y = parse("trainData.txt")
    outputClasses = [1, 2]
    colTypes = [True, True, True, True, True, True, True, False]
    hw = KNNClassifier(outputClasses, columntype=colTypes).fit(X, y)

    testX, testy = parse("testData.txt")
    solX, soly = parse("solution.txt")
    predictions = []
    misses = []
    diffs = []
    numCorrect = 0
    for i in range(len(testX)):
        prediction = hw.predict(testX[i])
        predictions.append(prediction)
        if prediction == testy[i]:
            numCorrect += 1
        else:
            misses.append(i)
        if prediction != soly[i]:
            diffs.append(i)

    score = hw.score(testX, testy)
    print(score)
    print("END TEST 3")
    print()

def test4():
    print("START TEST 4:")
    X, y = parse("hwData.txt")
    outputClasses = None
    colTypes = [True, True, True]
    hw = KNNClassifier(outputClasses, columntype=colTypes, weight_type="no_weight", regressionVals=True).fit(X, y)
    prediction = hw.predict(np.array([.5, .2]))
    print(prediction)
    print("END TEST 4")
    print()

def test5():
    print("START TEST 5:")
    X, y = parse("hwData.txt")
    outputClasses = None
    colTypes = [True, True, True]
    hw = KNNClassifier(outputClasses, columntype=colTypes, regressionVals=True).fit(X, y)
    prediction = hw.predict(np.array([.5, .2]))
    print(prediction)
    print("END TEST 5")
    print()

def test6():
    print("START TEST 6:")
    X, y = parse("trainData.txt")
    outputClasses = [1, 2]
    colTypes = [True, True, True, True, True, True, True, False]
    hw = KNNClassifier(outputClasses, columntype=colTypes, kList=list(range(1, 6))).fit(X, y)

    testX, testy = parse("testData.txt")
    score = hw.score(testX, testy)
    print(score)
    print("END TEST 6")
    print()

# test1()
test2()
test3()
# test4()
# test5()
test6()