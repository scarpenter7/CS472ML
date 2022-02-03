from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import random



class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, link_type='single'):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        if self.link_type == 'single':
            self.clusterTuples = self.singleLink(X, y)
        else:
            self.clusterTuples = self.completeLink(X, y)
        return self

    def singleLink(self, X, y):
        numClusters = len(X)
        clusters = [x for x in X]
        dMatrix = distance_matrix(X, X, p=2)
        np.fill_diagonal(dMatrix, np.inf)
        while numClusters != self.k:
            dMatrix, clusterIndices = self.singleLinkIter(dMatrix)

            # Join clusters
            cluster1 = clusters[clusterIndices[0]]
            cluster2 = clusters[clusterIndices[1]]
            newCluster = np.vstack((cluster1, cluster2))
            del clusters[np.max(clusterIndices)]
            del clusters[np.min(clusterIndices)]
            clusters.append(newCluster)
            numClusters -= 1

        if y is not None:
            ptLabelDict = {tuple(X[i]):y[i] for i in range(len(y))}
            clusterTuples = self.computeCentroidsWithLabels(clusters, ptLabelDict)
        else:
            clusterTuples = self.computeCentroidsNoLabels(clusters)

        return clusterTuples

    def singleLinkIter(self, dMatrix):
        clusterIndices = np.argwhere(dMatrix == np.min(dMatrix))[0]
        row1 = dMatrix[clusterIndices[0], :]
        row1[row1 == np.inf] = 0
        row2 = dMatrix[clusterIndices[1], :]
        row2[row2 == np.inf] = 0

        stackRows = np.vstack((row1, row2))
        newRow = np.amin(stackRows, axis=0)
        newRow = np.delete(newRow, clusterIndices)
        newRow0 = np.concatenate((newRow, np.array([np.inf])))
        newRowT = newRow[:, np.newaxis]

        dMatrix = np.delete(dMatrix, np.max(clusterIndices), 0)
        dMatrix = np.delete(dMatrix, np.max(clusterIndices), 1)
        dMatrix = np.delete(dMatrix, np.min(clusterIndices), 0)
        dMatrix = np.delete(dMatrix, np.min(clusterIndices), 1)

        dMatrix = np.hstack((dMatrix, newRowT))
        dMatrix = np.vstack((dMatrix, newRow0))
        return dMatrix, clusterIndices

    def completeLink(self, X, y):
        numClusters = len(X)
        clusters = [x for x in X]
        dMatrix = distance_matrix(X, X, p=2)
        np.fill_diagonal(dMatrix, np.inf)
        while numClusters != self.k:
            dMatrix, clusterIndices = self.completeLinkIter(dMatrix)

            # Join clusters
            cluster1 = clusters[clusterIndices[0]]
            cluster2 = clusters[clusterIndices[1]]
            newCluster = np.vstack((cluster1, cluster2))
            del clusters[np.max(clusterIndices)]
            del clusters[np.min(clusterIndices)]
            clusters.append(newCluster)
            numClusters -= 1

        if y is not None:
            ptLabelDict = {tuple(X[i]): y[i] for i in range(len(y))}
            clusterTuples = self.computeCentroidsWithLabels(clusters, ptLabelDict)
        else:
            clusterTuples = self.computeCentroidsNoLabels(clusters)

        return clusterTuples

    def completeLinkIter(self, dMatrix):
        clusterIndices = np.argwhere(dMatrix == np.min(dMatrix))[0]
        row1 = dMatrix[clusterIndices[0], :]
        row2 = dMatrix[clusterIndices[1], :]

        stackRows = np.vstack((row1, row2))
        newRow = np.amax(stackRows, axis=0)
        newRow = newRow[newRow != np.inf]
        newRow0 = np.concatenate((newRow, np.array([np.inf])))
        newRowT = newRow[:, np.newaxis]

        dMatrix = np.delete(dMatrix, np.max(clusterIndices), 0)
        dMatrix = np.delete(dMatrix, np.max(clusterIndices), 1)
        dMatrix = np.delete(dMatrix, np.min(clusterIndices), 0)
        dMatrix = np.delete(dMatrix, np.min(clusterIndices), 1)

        dMatrix = np.hstack((dMatrix, newRowT))
        dMatrix = np.vstack((dMatrix, newRow0))
        return dMatrix, clusterIndices

    def computeCentroidsWithLabels(self, clusters, ptLabelDict):
        pairs = []
        for cluster in clusters:
            if np.ndim(cluster) == 1:
                centroid = cluster
                labels = np.array([ptLabelDict.get(tuple(cluster))])
                labeledCluster = np.concatenate((cluster, labels))
                pairs.append((centroid, labeledCluster))
                continue
            centroid = np.mean(cluster, axis=0)
            labels = np.array([ptLabelDict.get(tuple(c)) for c in cluster])
            labels = labels[:, np.newaxis]
            labeledCluster = np.hstack((cluster, labels))
            pairs.append((centroid, labeledCluster))
        return pairs

    def computeCentroidsNoLabels(self, clusters):
        pairs = []
        for cluster in clusters:
            if np.ndim(cluster) == 1:
                centroid = cluster
                SSE = 0
                pairs.append((centroid, cluster, SSE))
                continue
            centroid = np.mean(cluster, axis=0)
            SSE = np.sum(np.array([(c - centroid)**2 for c in cluster]))
            pairs.append((centroid, cluster, SSE))
        return pairs

    def print_clusters(self):
        """
            Used for grading.
            print("{:d}\n".format(k))
            print("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                print(np.array2string(centroid,precision=4,separator=","))
                print("\n")
                print("{:d}\n".format(size of cluster))
                print("{:.4f}\n\n".format(SSE of cluster))
        """
        print("Number of Clusters = " + str(len(self.clusterTuples)))
        totalSSE = sum([c[2] for c in self.clusterTuples])
        print("Total SSE: " + str(totalSSE))
        print()
        for cluster in self.clusterTuples:
            centroid = cluster[0]
            clusterPts = cluster[1]
            SSE = cluster[2]
            if np.ndim(clusterPts) == 1:
                length = 1
            else:
                length = len(clusterPts)
            print("Centroid: " + str(centroid))
            print("Cluster Size: " + str(length))
            print("SSE: " + str(SSE))
            print()


class KMEANSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, debug=False):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
        self.numIters = 0

    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        # Initial centroids
        if self.debug:
            self.centroids = [X[i, :] for i in range(self.k)]
        else:
            indices = random.sample(range(len(X)), self.k)
            self.centroids = [X[i, :] for i in indices]
        oldCentroids = None
        while self.stoppingCriteria(oldCentroids):
            oldCentroids = self.epoch(X)
        self.clusterTuples = self.assembleClusterTuples()
        return self

    def epoch(self, X):
        oldCentroids = self.centroids.copy()

        # Sort data points into clusters
        self.clusters = [[] for _ in range(self.k)] # Clear out the clusters first
        for pt in X:
            dists = [np.linalg.norm(pt - self.centroids[i], ord=2) for i in range(self.k)]
            clusterIndex = np.argmin(dists)
            closestCluster = self.clusters[clusterIndex]
            closestCluster.append(list(pt))

        # Update centroids
        for i, cluster in enumerate(self.clusters):
            npCluster = np.array(cluster)
            newCentroid = np.mean(npCluster, axis=0)
            self.centroids[i] = newCentroid
        self.numIters += 1
        return oldCentroids

    def stoppingCriteria(self, oldCentroids):
        if self.numIters == 0:
            return True
        if np.sum(np.abs(np.array(oldCentroids) - np.array(self.centroids))) < 0.01: # convergence
            return False
        if self.numIters > 99: # max iterations
            return False
        return True

    def assembleClusterTuples(self):
        tuples = []
        for i, cluster in enumerate(self.clusters):
            npCluster = np.array(cluster)
            if np.ndim(npCluster) == 1:
                centroid = cluster
                SSE = 0
                self.clusterTuples.append((centroid, cluster, SSE))
                continue
            centroid = self.centroids[i]
            SSE = np.sum(np.array([(c - centroid) ** 2 for c in npCluster]))
            tuples.append((centroid, npCluster, SSE))
        return tuples

    def print_clusters(self):
        """
            Used for grading.
            print("{:d}\n".format(k))
            print("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                print(np.array2string(centroid,precision=4,separator=","))
                print("\n")
                print("{:d}\n".format(size of cluster))
                print("{:.4f}\n\n".format(SSE of cluster))
        """
        print("Number of Clusters = " + str(len(self.clusterTuples)))
        totalSSE = sum([c[2] for c in self.clusterTuples])
        print("Total SSE: " + str(totalSSE))
        print()
        for cluster in self.clusterTuples:
            centroid = cluster[0]
            clusterPts = cluster[1]
            SSE = cluster[2]
            if np.ndim(clusterPts) == 1:
                length = 1
            else:
                length = len(clusterPts)
            print("Centroid: " + str(centroid))
            print("Cluster Size: " + str(length))
            print("SSE: " + str(SSE))
            print()

# Tests:
def parse(filename):
    file = open(filename)
    lines = file.readlines()
    X = []
    for line in lines:
        nums = [float(num) for num in line.split(',')]
        X.append(nums)
    X = np.array(X)
    return X

def test1():
    print("START TEST 1:")
    X = np.array([[.8, .7],
                  [-.1, .2],
                  [.9, .8],
                  [0, .2],
                  [.2, .1]])

    y = np.array(['a', 'b', 'c', 'd', 'e'])

    hac = HACClustering(k=3).fit(X, y)
    for cluster in hac.clusterTuples:
        print("Centroid: " + str(cluster[0]))
        print("Cluster:")
        print(cluster[1])
        print()

    print("END TEST 1.")
    print()

def test2():
    print("START TEST 2:")
    X = np.array([[.8, .7],
                  [-.1, .2],
                  [.9, .8],
                  [0, .2],
                  [.2, .1]])

    hac = HACClustering(k=3).fit(X)
    for cluster in hac.clusterTuples:
        print("Centroid: " + str(cluster[0]))
        print("Cluster:")
        print(cluster[1])
        print()

    print("END TEST 2.")
    print()

def test3():
    print("START TEST 3:")
    X = np.array([[.8, .7],
                  [-.1, .2],
                  [.9, .8],
                  [0, .2],
                  [.2, .1]])

    hac = HACClustering(k=3, link_type='complete').fit(X)
    for cluster in hac.clusterTuples:
        print("Centroid: " + str(cluster[0]))
        print("Cluster:")
        print(cluster[1])
        print()

    print("END TEST 3.")
    print()

def test4():
    print("START TEST 4:")
    X = parse("abolone.txt")
    X = (X - np.min(X, 0))/(np.max(X, 0) - np.min(X, 0))
    hac = HACClustering(k=5).fit(X)
    hac.print_clusters()

    print("END TEST 4.")
    print()

def test5():
    print("START TEST 5:")
    X = parse("abolone.txt")
    X = (X - np.min(X, 0))/(np.max(X, 0) - np.min(X, 0))
    hac = HACClustering(k=5, link_type="complete").fit(X)
    hac.print_clusters()

    print("END TEST 5.")
    print()

def test6():
    print("START TEST 6:")
    X = np.array([[.9, .8],
                  [.2, .2],
                  [.7, .6],
                  [-.1, -.6],
                  [.5, .5]])

    y = np.array(['a', 'b', 'c', 'd', 'e'])

    kmeans = KMEANSClustering(k=2, debug=True).fit(X, y)
    for cluster in kmeans.clusterTuples:
        print("Centroid: " + str(cluster[0]))
        print("Cluster:")
        print(cluster[1])
        print()

    print("END TEST 6.")
    print()

def test7():
    print("START TEST 7:")
    X = np.array([[.9, .8],
                  [.2, .2],
                  [.7, .6],
                  [-.1, -.6],
                  [.5, .5]])

    y = np.array(['a', 'b', 'c', 'd', 'e'])

    kmeans = KMEANSClustering(k=2, debug=True).fit(X, y)
    kmeans.print_clusters()

    print("END TEST 7.")
    print()

def test8():
    print("START TEST 8:")
    X = parse("irisNums.txt")

    kmeans = HACClustering(k=3).fit(X)
    kmeans.print_clusters()

    print("END TEST 8.")
    print()

# test1()
# test2()
# test3()
# test4()
# test5()
# test6()
# test7()
test8()