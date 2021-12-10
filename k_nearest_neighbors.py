# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors=5, weights='uniform', metric='l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def _metric_func(self, x1, x2):
        return self._distance(x1, x2)

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y

    def _nearest_neighbors(self, X):
        """
        Get n nearest neighbors for each object in X.
        For each object in X returns the array of n
        closest objects in train set.

        Args:
            X(ndarray): objects
        Return:
            nearest_indices(ndarray): array of nearest
                                      objects indices
        """

        nearest_indices = np.zeros(shape=(X.shape[0], self.n_neighbors), dtype=np.int) - 1
        nearest_distances = np.zeros(shape=(X.shape[0], self.n_neighbors), dtype=np.int) - 1

        for i in range(X.shape[0]):
            if (i + 1) % 100 == 0:
                print("Object {} out of {} has been predicted".format(i + 1, X.shape[0]))
            distances = self._metric_func(X[i], self._X)
            index_order = np.argsort(distances)[:self.n_neighbors]
            nearest_indices[i] = index_order
            nearest_distances[i] = distances[index_order]

        return nearest_indices, nearest_distances

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        y = np.zeros(shape=(X.shape[0],))

        nearest_indices, nearest_distances = self._nearest_neighbors(X)
        nearest_labels = self._y[nearest_indices]

        for i in range(X.shape[0]):
            y[i] = np.argmax(np.bincount((nearest_labels[i])))

        return y
