from utils import *
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer


class CustomKMeans(object):
    def __init__(self, n_clusters=2, max_iter=100, threshold=0.001):
        """
        Parameters
        ----------
        n_clusters : int
        max_iter : int
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_iter_ = 0
        self.inertia_ = 0  # will contain the sum of squares error of each iteration
        self.threshold = threshold

    def fit(self, points):
        # cluster assignment
        n_points = len(points)
        centroids_labels = np.zeros(n_points, dtype=int)
        n_clusters = self.n_clusters
        max_iter = self.max_iter

        # initialize clusters_centers
        centroids = random_init(points, n_clusters)

        last_sse = 1

        for i in range(max_iter):
            # get cluster assignment for each point
            centroids_labels = get_closest(points, centroids)

            # update clusters_centers taking average point
            centroids = update_centroids(points, centroids_labels)

            # convergence test
            current_sse = int(compute_sse(points, centroids[centroids_labels]))
            if (abs(last_sse - current_sse) / last_sse) < self.threshold:
                break
            last_sse = current_sse

        self.inertia_ = current_sse
        self.labels_ = centroids_labels
        self.n_iter_ = i
