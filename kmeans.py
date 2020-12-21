from utils import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf(reviews):

    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=0.1
    )

    return tfidf.fit_transform(reviews)


class CustomKMeans(object):
    def __init__(self, n_clusters=2, max_iter=100, init='random', threshold_pct=None):
        """
        Parameters
        ----------
        n_clusters : int
        max_iter : int
        init : ['random', 'sharding']
        """
        assert(init in ["random", "sharding"])

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_iter_ = 0
        self.inertia_ = 0  # will contain the sum of squares error of each iteration
        self.threshold_pct = threshold_pct

        if init == 'random':
            self.init = random_init
        elif init == 'sharding':
            self.init = sharding_init

    def fit_predict(self, points):
        # cluster assignment
        n_points = len(points)
        centroids_labels = np.zeros(n_points, dtype=int)
        n_clusters = self.n_clusters
        max_iter = self.max_iter

        # initialize clusters_centers
        clusters_centers = self.init(points, n_clusters)

        for i in range(max_iter):
            # get cluster assignment for each point
            centroids_labels = get_closest(points, clusters_centers)

            # update clusters_centers taking average point
            clusters_centers = update_centroids(points, centroids_labels)

        self.inertia_ = int(compute_sse(points, clusters_centers[centroids_labels]))

        return centroids_labels
