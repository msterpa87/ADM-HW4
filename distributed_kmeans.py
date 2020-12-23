from kmeans import random_init
from operator import add


def distributed_kmeans(sc, points, n_clusters=2, max_iter=100, threshold_pct=0.001):
    """ Computes the distributed KMeans in pySpark

    Parameters
    ----------
    sc : a SparkContext instance
    points : a numpy of dim (n, d)
    n_clusters : int - the number of clusters
    max_iter : int - the maximum number of iterations
    threshold_pct : float - used to test for convergence

    Returns
    -------
        (labels, sse) : labels is a numpy of dim (n, 1), sse a float
    """
    def mapper_centroids(point):
        """ Mapper to compute closest centroid of each point

        Parameters
        ----------
        point : a tuple (point, 1), where point is a numpy of dim (n, d)

        Returns
        -------
        a tuple (i, (point, dist, 1)), where i and are resp. the index and
        distance of the closest centroid
        """
        diffs = ((point - centroids_broadcast.value) ** 2).sum(axis=1)
        i = diffs.argmin()
        dist = diffs.min()

        return i, (point, dist, 1)

    def mapper_sse(pair):
        """ Mapper to compute sum of squares error """
        _, (_, dist, _) = pair
        return 0, dist

    def reducer(pair1, pair2):
        """ Reducer to sum points in the same cluster

        Parameters
        ----------
        pair1 : a tuple (point1, dist1, x)
        pair2 : a tuple (point2, dist2, y)

        Returns
        -------
        a tuple (point, dist, x + y)
        this is used to computes the sum of points in the same cluster, tuples are
        aggregated by key, the third components will be the size of the cluster
        """
        x, u, s = pair1
        y, v, t = pair2

        return x + y, u + v, s + t

    # initialize RDD
    points_rdd = sc.parallelize(points)

    # sum of squares error
    last_sse = 0
    map_result = None

    # initialize clusters_centers
    centroids = random_init(points, n_clusters)

    # broadcast centroids to each node
    centroids_broadcast = sc.broadcast(centroids)

    for i in range(max_iter):
        map_result = points_rdd.map(mapper_centroids)

        # compute new centroids
        centroids = map_result.reduceByKey(reducer).collect()
        centroids = [x[0] / x[2] for _, x in centroids]

        # stopping criteria based on difference in sum of squares error
        current_sse = map_result.map(mapper_sse).reduceByKey(add).collect()[0][1]
        if abs(current_sse - last_sse) / current_sse < threshold_pct:
            break

        # save last_sse
        last_sse = current_sse

        # broadcast new centroids
        centroids_broadcast = sc.broadcast(centroids)

    # assigned clusters
    assigned_centroids = [x for x, _ in map_result.collect()]

    return assigned_centroids, int(current_sse)
