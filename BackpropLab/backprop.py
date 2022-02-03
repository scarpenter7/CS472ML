from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
import numpy as np
import random
import matplotlib.pyplot as plt


class MLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=.1, momentum=0, shuffle=True, hidden_layer_widths=None, debug=False, numOutputs=1,
                 validX=None, validy=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.numOutputs = numOutputs
        self.shuffle = shuffle
        self.debug = debug
        self.initial_weights = None
        self.momentum_weights = None
        self.A = []
        self.baseAccuracy = 0
        self.accuracies = []
        self.MSEs = []
        self.numInputs = None
        self.numLayers = hidden_layer_widths[0] + 1
        self.validX = validX
        self.validy = validy

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """

        # Add the bias column
        bias = np.ones((len(X), 1))
        X = np.hstack((X, bias))
        self.numInputs = len(X[0, :])

        # Initialize weights and momentum matrices
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        self.momentum_weights = self.initialize_momentum()

        # Fit
        self.numEpochs = 0
        while self.stoppingCriteria():
            self.performEpoch(X, y)
        return self

    def performEpoch(self, X, y):
        # Shuffle
        Xshuffle, yShuffle = self._shuffle_data(X, y)

        # Perform 1 epoch
        for i in range(len(yShuffle)):
            x = Xshuffle[i, :]
            if self.numOutputs > 1:
                y0 = yShuffle[i]
                target = np.array(self.convertEncoding(y0))
                a1 = self.evaluate(x)
                npa1 = np.array(a1)
                deltas = (target - npa1)*npa1*(1-npa1)
                self.backprop(deltas)
            else: # single output
                target = yShuffle[i]
                a1 = self.evaluate(x)[0]
                deltas = [(target - a1) * a1 * (1 - a1)]
                self.backprop(deltas)

            self.A = []
        self.numEpochs += 1

    def stoppingCriteria(self):
        if self.debug:
            if self.numEpochs < 10:
                return True
            return False
        if self.numEpochs == 0:
            return True
        accuracy = self.score(self.validX, self.validy)
        self.accuracies.append(accuracy)
        tolerance = 0.92
        self.baseAccuracy = accuracy
        if accuracy > tolerance:
            return False
        if self.numEpochs > 8 and\
            abs(accuracy - self.accuracies[-2]) < 0.01 and\
            abs(accuracy - self.accuracies[-3]) < 0.01:
            return False
        if self.numEpochs >= 50:
            return False
        return True

    def evaluate(self, x):
        inputs = x
        self.A.append(list(inputs))
        for layer in range(self.numLayers):
            currWeights = self.initial_weights[layer]
            numInNodes, numOutNodes = currWeights.shape

            # Compute Nets
            nets = self.computeNets(numOutNodes, inputs, currWeights)

            # Sigmoid function
            if layer == self.numLayers - 1:
                inputs = [1 / (1 + np.exp(-net)) for net in nets] # final output
            else:
                inputs = [1 / (1 + np.exp(-net)) for net in nets] + [1] # add bias

            # Store the computations
            self.A.append(inputs)

        # print(inputs)
        return inputs

    def evalTest(self, x):
        inputs = x
        for layer in range(self.numLayers):
            currWeights = self.initial_weights[layer]
            numInNodes, numOutNodes = currWeights.shape

            # Compute Nets
            nets = self.computeNets(numOutNodes, inputs, currWeights)

            # Sigmoid function
            if layer == self.numLayers - 1:
                inputs = [1 / (1 + np.exp(-net)) for net in nets]
            else:
                inputs = [1 / (1 + np.exp(-net)) for net in nets] + [1] # add bias

        # print(inputs)
        return inputs

    def convertEncoding(self, index):
        encoding = [0] * self.numOutputs
        encoding[int(index)] = 1
        return encoding

    def computeNets(self, numOutNodes, inputs, currWeights):
        nets = []
        for outNode in range(numOutNodes):
            net = np.dot(inputs, currWeights[:, outNode])
            nets.append(net)
        return nets

    def backprop(self, d1):
        deltas = [d1]
        for layer in range(self.numLayers):
            currDeltaLayer = deltas[-1]
            layerIndex = self.numLayers - layer - 1
            currWeights = self.initial_weights[layerIndex]
            rows, cols = currWeights.shape

            # Compute the weight adjustments
            adjustWeightsMatrix = np.zeros(currWeights.shape)
            A_row = self.A[layerIndex]
            for i in range(rows):
                for j in range(cols):
                    adjustWeightsMatrix[i, j] = self.adjustWeight(A_row[i], currDeltaLayer[j], layerIndex, i, j)
            self.momentum_weights[layerIndex] = adjustWeightsMatrix

            # Compute the next delta layer unless we are at the bottom layer
            if layerIndex != 0:
                deltaLayer = self.getNextDeltaLayer(A_row, currWeights, currDeltaLayer)
                deltas.append(deltaLayer)

            # Adjust weights
            self.initial_weights[layerIndex] = self.initial_weights[layerIndex] + adjustWeightsMatrix

    def adjustWeight(self, a, delta, layer, i, j):
        prevMovement = self.momentum_weights[layer][i, j]
        return self.lr * a * delta + self.momentum*prevMovement

    def getNextDeltaLayer(self, A_row, currentWeights, currDeltaLayer):
        deltaLayer = []
        for j, a in enumerate(A_row[:-1]):
            deltaWeights = currentWeights[j, :]
            currDelta = self.calcDelta(deltaWeights, currDeltaLayer, a)
            deltaLayer.append(currDelta)
        return deltaLayer

    def calcDelta(self, weights, deltas, a):
        return sum([deltas[k] * weights[k] for k in range(len(deltas))]) * a * (1 - a)

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        def activation(x):
            if len(x) == 1:
                if x[0] >= 0.5:
                    return 1
            else:
                index = np.argmax(x)
                return self.convertEncoding(index)

        predictions = [activation(self.evalTest(x0)) for x0 in X]
        return predictions

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        weights = []
        numRows = self.hidden_layer_widths[0] + 1
        inputLayers = [self.numInputs] + [self.hidden_layer_widths[1] + 1 for _ in range(self.hidden_layer_widths[0])]
        outputLayers = [self.hidden_layer_widths[1] for _ in range(self.hidden_layer_widths[0])] + [self.numOutputs]
        for i in range(numRows):
            if self.debug:
                weightNums = np.zeros(inputLayers[i] * outputLayers[i])
            else:
                weightNums = np.random.normal(0, 1, inputLayers[i] * outputLayers[i])
            wMatrix = np.array(weightNums).reshape((inputLayers[i], outputLayers[i]))
            weights.append(wMatrix)
        return weights

    def initialize_momentum(self):
        weights = []
        numRows = self.hidden_layer_widths[0] + 1
        inputLayers = [self.numInputs] + [self.hidden_layer_widths[1] + 1 for _ in range(self.hidden_layer_widths[0])]
        outputLayers = [self.hidden_layer_widths[1] for _ in range(self.hidden_layer_widths[0])] + [self.numOutputs]
        for i in range(numRows):
            weightNums = np.zeros(inputLayers[i] * outputLayers[i])
            wMatrix = np.array(weightNums).reshape((inputLayers[i], outputLayers[i]))
            weights.append(wMatrix)
        return weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        bias = np.ones((len(X), 1))
        X = np.hstack((X, bias))
        predictedVals = self.predict(X)

        errors = [1 for i in range(len(predictedVals))
                  if not np.allclose(np.array(self.convertEncoding(y[i])), np.array(predictedVals[i]))]
        self.MSEs.append(sum(errors)/len(predictedVals))
        return 1 - sum(errors) / len(predictedVals)

    def graphMSE(self):
        plt.plot(range(self.numEpochs), self.MSEs, label="lr = " + str(self.lr))
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.legend()

    def graphAccuracy(self):
        plt.plot(range(self.numEpochs), self.accuracies)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy %")

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        if self.shuffle:
            data = list(zip(X, y))
            random.shuffle(data)
            newX = [d[0] for d in data]
            newy = [d[1] for d in data]
            return np.array(newX), np.array(newy)
        return X, y

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.initial_weights

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

def shuffle_data(X, y):
    data = list(zip(X, y))
    random.shuffle(data)
    newX = [d[0] for d in data]
    newy = [d[1] for d in data]
    return np.array(newX), np.array(newy)

def test1():
    X, y = parse("debug.txt")

    myClassifier = MLPClassifier(hidden_layer_widths= [3, 3], debug=True, shuffle=False)
    fitted = myClassifier.fit(X, y)
    print(fitted.get_weights())
    print("finished")

def test2():
    X, y = parse("backpropHW.txt")

    initialWeights = [np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), np.array([[1.0], [1.0], [1.0]])]

    myClassifier = MLPClassifier(hidden_layer_widths= [1, 2], debug=True, lr=1, shuffle=False)
    fitted = myClassifier.fit(X, y, initial_weights=initialWeights)
    print(fitted.get_weights())
    print("finished")

def test3():
    print("START TEST 3:")
    X, y = parse("debug.txt")

    initialWeights = [np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]]),
                      np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])]

    myClassifier = MLPClassifier(hidden_layer_widths=[1, 4], debug=True, shuffle=False, momentum=0.5)
    fitted = myClassifier.fit(X, y, initial_weights=initialWeights)
    print("Epochs: " + str(fitted.numEpochs))
    print("Layer 1:")
    print(fitted.get_weights()[0])
    print()
    print("Layer 2:")
    print(fitted.get_weights()[1])
    print("END TEST 3:")
    print()

def test4():
    print("START TEST 4:")
    X, y = parse("debug.txt")

    initialWeights = [np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                      np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])]

    myClassifier = MLPClassifier(hidden_layer_widths=[1, 4], debug=True, shuffle=False, momentum=0.5, numOutputs=2)
    fitted = myClassifier.fit(X, y, initial_weights=initialWeights)
    print("Epochs: " + str(fitted.numEpochs))
    print("Layer 1:")
    print(fitted.get_weights()[0])
    print()
    print("Layer 2:")
    print(fitted.get_weights()[1])
    print("END TEST 4")
    print()

def test5():
    print("START TEST 5:")
    X, y = parse("evaluation.txt")

    initialWeights = [np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]]),
                      np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])]

    myClassifier = MLPClassifier(hidden_layer_widths=[1, 4], debug=True, shuffle=False, momentum=0.5)
    fitted = myClassifier.fit(X, y, initial_weights=initialWeights)
    print("Epochs: " + str(fitted.numEpochs))
    print("Layer 1:")
    print(fitted.get_weights()[0])
    print()
    print("Layer 2:")
    print(fitted.get_weights()[1])
    print("END TEST 5")
    print()

def test6():
    print("START TEST 6:")
    X, y = parse("evaluation.txt")

    initialWeights = [np.array([[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]]),
                      np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])]

    myClassifier = MLPClassifier(hidden_layer_widths=[1, 4], debug=True, shuffle=False, momentum=0.5, numOutputs=2)
    fitted = myClassifier.fit(X, y, initial_weights=initialWeights)
    print("Epochs: " + str(fitted.numEpochs))
    print("Layer 1:")
    print(fitted.get_weights()[0])
    print()
    print("Layer 2:")
    print(fitted.get_weights()[1])
    print("END TEST 6")
    print()

def test7():
    print("START TEST 7:")
    X, y = parse("irisNums.txt")
    shuffleX, shuffley = shuffle_data(X, y)

    trainX = shuffleX[:int(.8*len(X)), :]
    trainy = shuffley[:int(.8*len(y))]

    validX = trainX[:int(.15 * len(X)), :]
    validy = trainy[:int(.15 * len(y))]

    testX = shuffleX[int(.8 * len(X)):, :]
    testy = shuffley[int(.8 * len(y)):]


    myClassifier = MLPClassifier(hidden_layer_widths=[1, 8], momentum=0.5, numOutputs=3, validX=validX, validy=validy)
    fitted = myClassifier.fit(trainX, trainy)
    print("Epochs: " + str(fitted.numEpochs))

    score = fitted.score(testX, testy)
    print("Score: ")
    print(score)
    fitted.graphMSE()
    plt.show()
    fitted.graphAccuracy()
    plt.show()
    print("END TEST 7")
    print()

def testlr(lr):
    X, y = parse("vowelsNums.txt")
    shuffleX, shuffley = shuffle_data(X, y)

    trainX = shuffleX[:int(.8*len(X)), :]
    trainy = shuffley[:int(.8*len(y))]

    validX = trainX[:int(.15 * len(X)), :]
    validy = trainy[:int(.15 * len(y))]

    testX = shuffleX[int(.8 * len(X)):, :]
    testy = shuffley[int(.8 * len(y)):]


    myClassifier = MLPClassifier(lr=lr, hidden_layer_widths=[1, 26], momentum=0.5, numOutputs=11, validX=validX, validy=validy)
    fitted = myClassifier.fit(trainX, trainy)
    print("Epochs: " + str(fitted.numEpochs))

    score = fitted.score(testX, testy)
    print("Score: ")
    print(score)
    fitted.graphMSE()

def test8():
    print("START TEST 8:")
    lrs = [0.02, 0.08, 0.2, 0.5, 1.0]
    for lr in lrs:
        print("Learning rate: " + str(lr))
        testlr(lr)
    plt.show()
    print("END TEST 8")
    print()

def test9():
    X, y = [], [] #Fix

    trainX = X[:int(.8 * len(X)), :]
    trainy = y[:int(.8 * len(y))]

    validX = trainX[:int(.15 * len(X)), :]
    validy = trainy[:int(.15 * len(y))]

    testX = X[int(.8 * len(X)):, :]
    testy = y[int(.8 * len(y)):]

    # Train on voting dataset
    # Learning rates
    print("Different learning rates")
    for lr in [.1 * i for i in range(1, 6)]:
        vote = MLPClassifier(lr=lr)
        fitted = vote.fit(trainX, trainy)
        score = fitted.score(testX, testy)
        print(score)

    # nodes and layers
    print()
    print("Different nodes and layers")
    for node in [3 * i for i in range(1, 6)]:
        vote = MLPClassifier(random_state=0, hidden_layer_sizes=(node, node))
        fitted = vote.fit(trainX, trainy)
        score = fitted.score(testX, testy)
        print(score)

    # Activation functions
    print()
    print("Different activation functions")
    for func in ['identity', 'logistic', 'tanh', 'relu']:
        vote = MLPClassifier(max_iter=100, random_state=0, activation=func)
        fitted = vote.fit(trainX, trainy)
        score = fitted.score(testX, testy)
        print(score)

    # regularization and params
    print()
    print("Different regularization and params")
    for alpha in [.0001 * i for i in range(1, 6)]:
        vote = MLPClassifier(random_state=0, alpha=lr)
        fitted = vote.fit(trainX, trainy)
        score = fitted.score(testX, testy)
        print(score)

    # Momentum
    print()
    print("Different nodes and momentums")
    for mom in [.15 * i for i in range(1, 8)]:
        vote = MLPClassifier(random_state=0, momentum=mom)
        fitted = vote.fit(trainX, trainy)
        score = fitted.score(testX, testy)
        print(score)

    # early stopping
    print()
    print("Test early stopping")
    for stop in [False, True]:
        vote = MLPClassifier(random_state=0, early_stopping=stop)
        fitted = vote.fit(trainX, trainy)
        score = fitted.score(testX, testy)
        print(score)

# Tests
# test1()
# test2()
# test3()
# test4()
# test5()
# test6()
# test7()
# test8()