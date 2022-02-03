from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

"""
Use standard information gain as your basic attribute evaluation metric.

(Note that normal ID3 would usually augment information gain with gain ratio or some other mechanism
to penalize statistically insignificant attribute splits. Otherwise, even with approaches like pruning below,
the SSE type of overfit could still hurt us.)

You are welcome to create other classes and/or functions in addition to the ones provided below.
(e.g. If you build out a tree structure, you might create a node class).

It is a good idea to use a simple data set (like the lenses data or the pizza homework),
which you can check by hand, to test your algorithm to make sure that it is working correctly.
"""

class Tree():

    def __init__(self, X, y, info, featuresLeft):
        self.root = Node("root", featuresLeft, info, X, y)
        self.currLayer = [self.root]


class Node():

    def __init__(self, featureIndex, featuresLeft, info, X, y):
        self.featureIndex = featureIndex
        self.info = info
        self.X = X
        self.y = y
        self.featuresLeft = featuresLeft
        self.children = []

    def calcPurity(self):
        if len(self.y) == 0:
            # Vacuously pure
            return 1, None
        mode = st.mode(self.y)
        purity = sum([1 for target in self.y if target == mode]) / len(self.y)
        return purity, int(mode)

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,counts=None):
        """ Initialize class with chosen hyperparameters.
        Args:
        Optional Args (Args we think will make your life easier):
            counts: A list of Ints that tell you how many types of each feature there are
        Example:
            DT  = DTClassifier()
            or
            DT = DTClassifier(count = [2,3,2,2])
            Dataset =
            [[0,1,0,0],
            [1,2,1,1],
            [0,1,1,0],
            [1,2,0,1],
            [0,0,1,1]]

        """
        self.counts = counts
        self.gainz = []

    def fit(self, X, y):
        """ Fit the data; Make the Decision tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 1D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # Initialize Tree
        rootInfo = self.calcInfo(y)
        featuresLeft = list(range(len(X[0, :])))
        self.decisionTree = Tree(X, y, rootInfo, featuresLeft)
        while True:
            newLayer = []
            for node in self.decisionTree.currLayer:
                if not self.nodeStoppingCriteria(node):
                    continue
                currX = node.X
                curry = node.y
                info = self.calcInfo(curry)
                featuresLeft = node.featuresLeft
                newNodes = self.split(currX, curry, featuresLeft, info)
                # add the new nodes to the tree
                node.children = newNodes
                newLayer.extend(newNodes)
            if len(newLayer) == 0:
                break
            self.decisionTree.currLayer = newLayer
        return self

    def nodeStoppingCriteria(self, node):
        purity, mode = node.calcPurity()
        if purity > 0.9:
            return False
        if len(node.featuresLeft) == 0:
            return False
        return True

    def split(self, X, y, featuresLeft, info):
        infos = {self.calcAttributeInfo(i, X, y): i for i in featuresLeft}
        minInfo = min(infos.keys())
        optimalFeatureIndex = infos.get(minInfo)
        numFeatureClasses = self.counts[optimalFeatureIndex]
        featuresLeft.remove(optimalFeatureIndex)
        newNodes = []
        for featureClass in range(numFeatureClasses):
            # Grab all the rows in X and y that match the current class of the optimal feature
            rowIndices = [i for i, x in enumerate(X[:, optimalFeatureIndex]) if x == featureClass]
            newX = X[rowIndices, :]
            newy = y[rowIndices]
            newNode = Node(optimalFeatureIndex, featuresLeft.copy(), minInfo, newX, newy)
            newNodes.append(newNode)
        gain = info - minInfo
        self.gainz.append(gain)
        return newNodes

    def calcInfo(self, y):
        # convert each target to a 1d numpy array with one hot encoding
        targets = np.array([np.array(self.convertEncoding(y[i], self.counts[-1])) for i in range(len(y))])
        probabilities = np.sum(targets, axis=0) / len(targets)
        info = - sum([probabilities[i] * np.log2(probabilities[i]) for i in range(self.counts[-1]) if probabilities[i] != 0])
        return info

    def calcAttributeInfo(self, featureIndex, X, y):
        numOutputs = self.counts[featureIndex]
        data = X[:, featureIndex]
        oneHots = np.array([np.array(self.convertEncoding(data[i], self.counts[featureIndex])) for i in range(len(data))])
        probabilities = np.sum(oneHots, axis=0) / len(data)
        yBlocks = []
        for output in range(numOutputs):
            # grab the y targets that match the current output class
            indices = [i for i, x in enumerate(data) if x == output]
            yBlock = np.array([y[i] for i in indices])
            yBlocks.append(yBlock)

        info = sum([probabilities[i] * self.calcInfo(yBlocks[i]) for i in range(numOutputs) if probabilities[i] != 0])
        return info

    def convertEncoding(self, index, numOutputs):
        encoding = [0] * numOutputs
        encoding[int(index)] = 1
        return encoding

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        predictions = [self.predictPt(row) for row in X]
        return predictions


    def predictPt(self, row):
        currNode = self.decisionTree.root
        while True:
            nextLayer = currNode.children
            nextFeature = nextLayer[0].featureIndex
            childIndex = int(row[nextFeature])
            currNode = nextLayer[childIndex]
            if len(currNode.children) == 0:
                purity, mode = currNode.calcPurity()
                return mode




    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 1D numpy array of the targets
        """
        predictions = np.array(self.predict(X))
        accuracy = sum([1 for i, p in enumerate(y) if predictions[i] == int(p)]) / len(predictions)
        return accuracy

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

def translate(list):
    # 0 --> 1
    # 1 --> 2
    # 2 --> 0
    newList = []
    for l in list:
        if l == 0:
            newList.append(1)
        elif l == 1:
            newList.append(2)
        else:
            newList.append(0)
    return newList

# Tests

def test1():
    print("START TEST 1:")
    X, y = parse("pizzaNums.txt")

    pizza = DTClassifier(counts=[2, 3, 2, 3]).fit(X, y)
    print("END TEST 1")
    print()

def test2():
    print("START TEST 2:")
    X, y = parse("lensesTrainNums.txt")

    testX, testy = parse("lensesPredictNums.txt")

    lenses = DTClassifier(counts=[3, 2, 2, 2, 3]).fit(X, y)
    print(lenses.gainz)
    predictions = lenses.predict(testX)
    translated = translate(predictions)
    # print(translated)
    accuracy = lenses.score(testX, testy)

    print(accuracy)
    print("END TEST 2")
    print()

def test3():
    print("START TEST 3:")
    X, y = parse("zoo.txt")
    print(y)
    testX, testy = parse("zoo.txt")

    zoo = DTClassifier(counts=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 2, 2, 2, 7]).fit(X, y)
    predictions = zoo.predict(testX)
    accuracy = zoo.score(testX, testy)

    print(accuracy)
    print("END TEST 3")
    print()

def test4():
    print("START TEST 4:")
    X, y = parse("votingMissing.txt")
    print(y)
    #testX, testy = parse("zoo.txt")

    zoo = DTClassifier(counts=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]).fit(X, y)
    #predictions = zoo.predict(testX)
    #accuracy = zoo.score(testX, testy)

    #print(accuracy)
    print("END TEST 4")
    print()

# test1()
# test2()
# test3()
test4()



