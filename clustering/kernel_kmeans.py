from __future__ import division
from random import randrange, randint, random
from collections import defaultdict

import numpy     as np
import itertools as it


class KMeansKernel:
    """
    Batch kmeans clustering of a set of vectors using
    a linear kernel as measure of distance metric.
    
    :param num_clusters   : number of clusters to compute 
    :param num_iterations : number of iterations to perform 
    :param initialization : cluster initialization algorithm to use defaults to 
                            "random", using kmeans++ is recommended
    """

    def __init__(self, num_clusters=2, num_iterations=100, \
                 initialization='random'):
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        self.epsilon = 1e-15
        print "value of eps :: ", self.epsilon
        assert (initialization == "random" or initialization == "kmeans++"), \
            "Unknown initialization scheme"
        self.initialization = initialization

    def _cluster_compactness(self, ci, refresh=False, cj=None):
        """
        compute the cluster compactness of the ith cluster 
        :param ci : cluster label 
        :param cj : optional cluster label, if specified 
                    the compactness measure of two clusters combined will be
                    computed, refresh will be ignored 
        :param refresh : ignore the memoized value and compute the 
                         new fresh value
        """
        try:
            self._compactness
        except AttributeError, e:
            self._compactness = dict()

        if cj:
            refresh = True
        else:
            cj = ci

        if not refresh and self._compactness.has_key(ci):
            return self._compactness[ci]

        num_ci = len(self._clusters_old[ci])
        num_cj = len(self._clusters_old[cj])
        # no points assigned to this cluster return 0 as measure of compactness
        if num_ci == 0 or num_cj == 0:
            if num_ci == 0: self._compactness[ci] = 0
            if num_cj == 0: self._compactness[cj] = 0
            return 0

        mc = 0
        for i, ai in self._clusters_old[ci]:
            for j, aj in self._clusters_old[cj]:
                mc += np.dot(ai, aj)

        mc = mc / (num_ci * num_cj)
        # cache only if cj is not specified 
        if cj == ci: self._compactness[ci] = mc
        return mc

    def __centroid_distance(self, ci, cj):
        """
        compute the kernelized distance between centroids of 
        cluster center ci and cj
        :param ci : cluster label for ith cluster
        :param cj : cluster label for jth cluster
        """
        return np.power(self._cluster_compactness(ci), 2) + np.power \
            (self._cluster_compactness(cj), 2) - 2 * self._cluster_compactness(ci, cj)

    def __distance(self, x, ci):
        """
        compute the kernelized distance between vector x and a cluster center ci
        :param x : datapoint vector
        :param ci : the cluster label for the ith cluster
        """
        num_c = len(self._clusters_old[ci])
        return np.dot(x, x) - (2 * sum([np.dot(x, ai) for (l, ai) in self._clusters_old[ci]]) / num_c) \
               + self._cluster_compactness(ci)

    def kmeans_init(self):
        """
        initialize the cluster centers based on k_means++ algorithm
        :return : set of initialized cluster means  
        """
        self._cluster_centers = []
        # initiliaze the first cluster centroid randomly in space
        first_label = randint(0, self.num_samples - 1)
        self._cluster_centers.append((first_label, self.X[first_label]))

        for choice in range(1, self.num_clusters):
            density_vector = np.zeros(self.num_samples)
            for label, point in enumerate(self.X):
                min_dis = min([np.linalg.norm(point - c) for l, c in self._cluster_centers])
                density_vector[label] = min_dis * min_dis
            # choose a point from the probability mass function introduced by 
            # D(x) * D(x) where D(x) is distance of point x from its nearest cluster
            density_vector /= np.sum(density_vector)
            density_vector = zip(range(self.num_samples), density_vector)
            density_vector = sorted(density_vector, key=lambda e: e[1])
            rand_x, sum_d, choosen_label = random(), 0, 0

            for l, d in density_vector:
                if sum_d >= rand_x:
                    choosen_label = l
                    break
                else:
                    sum_d += d
            self._cluster_centers.append((choosen_label, self.X[choosen_label]))

    def fit(self, X, Y=None):
        """
        train the clustering algorithm on the dataset
        :param X : numpy ndarray of shape (num_samples, num_features) or 
                   a list with similar shape 
        """
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            raise RuntimeError("X must be a list or a numpy ndarray")
        if isinstance(X, list):
            self.X = np.asarray(X)
        else:
            self.X = X
        (self.num_samples, self.num_features) = self.X.shape

        feature_value_ranges = []
        for f_index in range(self.num_features):
            feature_values = self.X.T[f_index]
            feature_min, feature_max = np.min(feature_values), np.max(feature_values)
            feature_value_ranges.append((feature_min, feature_max))

        self.cluster_labels = range(self.num_clusters)
        self._labels = {i: 0 for i in range(self.num_samples)}

        if self.initialization == "random":
            initial_centers = [randint(0, self.num_samples - 1) for i in range(self.num_clusters)]
            self._cluster_centers = [(k, self.X[k]) for k in initial_centers]
        else:
            # use kmeans++ algorithm to init the cluster centers
            self.kmeans_init()

        self._clusters = {i: [self._cluster_centers[i]] for i in range(self.num_clusters)}
        self.iterations = 1
        while not self.__converged():

            self._cluster_centers_old = self._cluster_centers[:]
            self._clusters_old = dict(self._clusters)
            self._clusters = defaultdict(list)

            for cluster_label in self._clusters_old:
                self._cluster_compactness(cluster_label, True)

            # assign points to the clusters 
            for data_index, data_point in enumerate(self.X):
                min_dis, label = min([(abs(self.__distance(data_point, label)), label) for label \
                                      in self._clusters_old], key=lambda e: e[0])
                self._labels[data_index] = label
                self._clusters[label].append((data_index, data_point))

            # recompute the means of the clusters
            for (cluster_label, points) in self._clusters.items():
                cluster_center = np.zeros(self.num_features)
                num_points = 0
                for l, p in points:
                    cluster_center = np.add(cluster_center, p)
                    num_points += 1

                cluster_center /= num_points
                self._cluster_centers[cluster_label] = cluster_center
            self.iterations += 1

        print "Number of iterations ran ::", self.iterations
        print "D-B index for clustering ::", self.__quality()

    def predict(self, X, Y=None):
        """
        predict the class labels for the data_points in test set X
        :param X : (num_samples, num_features) shape numpy array or similar shape list
        """
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            raise RuntimeError("X must be a list or a numpy ndarray")
        labels = []
        for data_point in X:
            min_dis, label = min([(abs(self.__distance(data_point, label)), label) for label \
                                  in self._clusters], key=lambda e: e[0])
            labels.append(label)
        if not Y:
            return labels
        else:
            return [(y, label) for (y, label) in it.izip(Y, labels)]

    def __converged(self):
        """
        covergence condition for the k-means, intuitively check 
        the number of cluster re-assignments and infer convergence if 
        cluster reassignments are less than a threshold
        """
        if self.iterations == 1: return False
        if self.iterations >= self.num_iterations: return True
        point_allocation_old, point_allocation = {}, {}

        for c, elements in self._clusters_old.items():
            for l, e in elements:
                point_allocation_old[l] = c

        for c, elements in self._clusters.items():
            for l, e in elements:
                point_allocation[l] = c

        movements = 0
        for e, c in point_allocation.items():
            if not point_allocation_old.has_key(e):
                movements += 1
                continue
            c_old = point_allocation_old[e]
            if c_old != c: movements += 1

        print "cluster point movements ::", movements
        if movements <= 1:
            return True
        else:
            return False

    def __quality(self):
        """
        compute the quality of clustering based on Davies-Bouldin
        index, lower values indicate better clustering
        reference : https://en.wikipedia.org/wiki/Davies-Bouldin_index
        """
        # assign old clusters to new one to measure quality
        self._clusters_old = self._clusters

        inter_cluster_distances = np.zeros(self.num_clusters)
        intra_cluster_distances = np.zeros(shape=(self.num_clusters, self.num_clusters))
        R = np.zeros(shape=(self.num_clusters, self.num_clusters))
        D = np.zeros(self.num_clusters)
        for ci, elements in self._clusters.items():
            num_ci = len(elements)
            inter_cluster_distances[ci] = sum([self.__distance(e, ci) for l, e in elements]) / num_ci

        for ci in self._clusters:
            for cj in self._clusters:
                if ci >= cj: continue
                intra_cluster_distances[ci, cj] = self._cluster_compactness(ci, cj)
                intra_cluster_distances[cj, ci] = intra_cluster_distances[ci, cj]
                R[ci, cj] = (inter_cluster_distances[ci] + inter_cluster_distances[cj]) \
                            / intra_cluster_distances[ci, cj]
                R[cj, ci] = R[ci, cj]

        for ci in self._clusters: D[ci] = max(R[ci, :])
        return sum(D) / self.num_clusters
