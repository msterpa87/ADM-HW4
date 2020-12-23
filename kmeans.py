import numpy as np
from numpy.linalg import norm


def distance(points, centroids):
    """ Returns the vector of distances of points to respective centroids

    Parameters
    ----------
    points : numpy (n, 1, d)
    centroids : numpy (1, k, d)
        centroids[i] = closest centroid to point[i]
    Returns
    -------
    distances : numpy of dim n
        the vector of distance of each point from its centroid
    """
    return ((points - centroids) ** 2).sum(axis=1)


def get_closest(points, centroids):
    """

    :param points: vector of dim (n, 1, d)
    :param centroids: vector of dim (1, k, d)
    :return: indices of the centroids closest to each point
    """
    return ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def average(points):
    """ Average point of the given set of points """
    return np.average(points, axis=0)


def compute_sse(points, centroids):
    """

    Parameters
    ----------
    points : numpy of dim (n, d)
    centroids : numpy of dim (n, d)
        centroids[i] = centroid of points[i]
    Returns
    -------
        float : sum of squares error
    """
    return distance(points, centroids).sum()


def update_centroids(points, clusters, n_centroids):
    x = np.array([(points[clusters == i]).mean(axis=0) for i in range(n_centroids)])
    return np.nan_to_num(x)


def clusters_size(clusters):
    return np.unique(clusters, return_counts=True)[1]


def random_init(points, n_clusters):
    """ Assign centroids by randomly sampling among points """
    random_indices = np.random.choice(range(len(points)), n_clusters, replace=False)

    return points[random_indices]


class CustomKMeans(object):
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4):
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
        self.tol = tol

    def fit(self, points):
        # cluster assignment
        n_points = len(points)
        centroids_labels = np.zeros(n_points, dtype=int)
        n_clusters = self.n_clusters
        max_iter = self.max_iter

        # initialize clusters_centers
        centroids = random_init(points, n_clusters)

        last_centroids = centroids
        best_sse = np.inf
        best_centroids = last_centroids

        for i in range(max_iter):
            # get cluster assignment for each point
            centroids_labels = get_closest(points, centroids)

            # update clusters_centers taking average point
            centroids = update_centroids(points, centroids_labels, n_clusters)

            # keep best centroids based on sum of squares error
            current_sse = int(compute_sse(points, centroids[centroids_labels]))
            if current_sse < best_sse:
                best_sse = current_sse
                best_centroids = centroids

            # convergence criteria using Frobenius norm
            center_shift = ((centroids - last_centroids) ** 2).sum()
            if center_shift < self.tol:
                break
            last_centroids = centroids

        # update instance variable with best data found so far
        self.inertia_ = best_sse
        self.labels_ = get_closest(points, best_centroids)
        self.n_iter_ = i
