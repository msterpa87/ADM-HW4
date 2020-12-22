from pyspark import SparkContext
from utils import random_init
from operator import add


def distributed_kmeans(sc, points, n_clusters=2, max_iter=100, threshold_pct=0.001):
    assert(max_iter > 0 and n_clusters > 1)
    # initialize clusters_centers
    centroids = random_init(points, n_clusters)
    # broadcast centroids to each node
    centroids_broadcast = sc.broadcast(centroids)

    def mapper_centroids(point):
        diffs = ((point - centroids_broadcast.value) ** 2).sum(axis=1)
        i = diffs.argmin()
        v = diffs.min()

        return i, (point, v, 1)

    def mapper_sse(pair):
        _, (_, dist, _) = pair

        return 0, dist

    def reducer(pair1, pair2):
        x, u, s = pair1
        y, v, t = pair2

        return x + y, u + v, s + t

    # initialize RDD
    points_rdd = sc.parallelize(points)
    # sum of squares error
    last_sse = 0
    map_result = None

    for i in range(max_iter):
        map_result = points_rdd.map(mapper_centroids)

        # compute new centroids
        centroids = map_result.reduceByKey(reducer).collect()
        centroids = [x[0] / x[2] for _, x in centroids]

        current_sse = map_result.map(mapper_sse).reduceByKey(add).collect()[0][1]

        # print("iter = {} - sse = {}".format(i, current_sse))

        if abs(current_sse - last_sse) / current_sse < threshold_pct:
            break

        # save last_sse
        last_sse = current_sse

        # broadcast new centroids
        centroids_broadcast = sc.broadcast(centroids)

    # assigned clusters
    assigned_centroids = [x for x, _ in map_result.collect()]

    return assigned_centroids
