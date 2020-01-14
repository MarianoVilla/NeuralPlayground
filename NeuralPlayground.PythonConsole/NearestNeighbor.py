import numpy as np
#A first approach to image recognition is a nearest neighbor classifier.
#This simple compares the given input to every other image it has trained with,
#and uses the label of the nearest, defining closenes by pixel-wise arithmetic.
#In our case, subtracting the matrices being compared, then summing all the elements in the resulting one.
#A generalization is the k-NearestNeighbor, which does the but using the k nearest neghbors.
#NearestNeighbor is indeed a special case of k-NearestNeghbor, i.e.: 1-NearestNeighbor.
class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """X is N x D where each row is an example. Y is 1-dimension of size N. """
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """X is N x D where each row is an example we wish to predict label for. """
        num_test = X.shape[0]

        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred

