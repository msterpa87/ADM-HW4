from kmeans import *
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")


def plot_variance(tfidf):
    """ Plots the variance by number of components in SVD """
    variance_list = []

    n = tfidf.shape[1]

    for i in range(2, n):
        svd = TruncatedSVD(n_components=i)
        svd.fit_transform(tfidf)
        variance = svd.explained_variance_.sum()
        variance_list.append(variance)

    components = np.arange(2, len(variance_list) + 2)
    variance_list = np.array(variance_list)
    x_below_idx = variance_list < 0.6
    x_above_idx = variance_list >= 0.6

    x_below = np.where(x_below_idx)[0]
    x_above = np.where(x_above_idx)[0]

    y_below = variance_list[x_below]
    y_above = variance_list[x_above]

    fig, ax = plt.subplots()
    ax.scatter(x_below + 2, y_below, marker='o')
    ax.scatter(x_above + 2, y_above, marker='o', color='red')
    ax.set_xticks(components)
    ax.axhline(0.6, linestyle='--', linewidth=1, color='red')
    ax.set(xlabel="Number of Components",
           ylabel="Variance",
           title="Explained Variance after SVD")


def kmeans_simulation(points, kmeans_class=CustomKMeans, max_centroids=20, max_iter=50, init='random'):
    """

    Parameters
    ----------
    points : a numpy of dim (n, d)
    kmeans_class : a class that implements fit_predict() and has attributes n_iter_ and intertia_
                   like the sklearn implementation
    max_centroids : int
    max_iter : int
    init : either "random" or "sharding"
        specify how to randomize the initial centroids

    Returns
    -------
    dict {i: {n_iter: int, sse: float, time: float}
        a dictionary providing information about the performance of KMeans
        on different number of centroids
    """
    kmeans_runs = {}

    for k in range(2, max_centroids + 1):
        tic = time.time()
        print(k, end=' ')

        custom_kmeans = kmeans_class(max_iter=max_iter, n_clusters=k)
        custom_kmeans.fit(points)

        toc = time.time()

        sse = int(custom_kmeans.inertia_)
        n_iter = custom_kmeans.n_iter_
        time_ = round(toc - tic, 2)

        kmeans_runs[k] = dict(n_iter=n_iter, sse=sse, time=time_)

    with open("simulation_data.pkl", "wb") as f:
        pickle.dump(kmeans_runs, f)

    return kmeans_runs


def plot_sse(data):
    x = list(data.keys())
    y = np.array([d['sse'] for d in data.values()])
    x_best = np.where((y[1:] - y[:-1]) / y[:-1] > 0)[0][0]
    y_best = y[x_best]
    x_best = x_best + 2  # x starts from 2

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.plot(x_best, y_best, marker='o', color='red')
    ax.set_xticks(x)
    ax.annotate("Best value?",
                xy=(x_best, y_best),
                xytext=(x_best + 0.1, y_best + 1e4))
    plt.title("SSE by Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squares Error")


def plot_sse_comparison(sklearn_data, custom_data):
    sklearn_df = pd.DataFrame.from_dict(sklearn_data, orient='index')
    custom_df = pd.DataFrame.from_dict(custom_data, orient='index')
    x = list(sklearn_df.index)
    plt.xticks(x)
    plt.plot(x, sklearn_df["sse"], marker='o')
    plt.plot(x, custom_df["sse"], marker='o')
    plt.legend(["sklearn kmeans", "custom kmeans"])
    plt.title("Sklearn vs Custom KMeans SSE")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squares Error")


def plot_performance(sklearn_data, custom_data):
    sklearn_df = pd.DataFrame.from_dict(sklearn_data, orient='index')
    custom_df = pd.DataFrame.from_dict(custom_data, orient='index')
    x = list(sklearn_df.index)
    plt.xticks(x)
    plt.plot(x, sklearn_df["time"], marker='o')
    plt.plot(x, custom_df["time"], marker='o')
    plt.legend(["sklearn kmeans", "custom kmeans"])
    plt.title("Sklearn vs Custom KMeans (Time)")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Time (s)")


def plot_distortion(data):
    x = list(data.keys())
    y = [d['distortion'] for d in data.values()]
    plt.plot(x, y)
    plt.title("Distortion by Number of Clusters")
    plt.xlabel("Number of Centroids")
    plt.ylabel("Distortion")