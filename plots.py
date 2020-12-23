from kmeans import *
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud, STOPWORDS
from utils import cluster_text_as_dict

plt.style.use("seaborn-whitegrid")


def plot_variance(tfidf, skip=1):
    """ Plots the variance by number of components in SVD """
    n = tfidf.shape[1]
    variance_list = []
    components = []

    for i in range(2, n, skip):
        svd = TruncatedSVD(n_components=i)
        svd.fit_transform(tfidf)
        variance = svd.explained_variance_.sum()
        variance_list.append(variance)
        components.append(i)

    components = np.array(components)
    variance_list = np.array(variance_list)

    with open("variance_data.pkl", "wb") as f:
        pickle.dump([components, variance_list], f)

    x_below_idx = variance_list < 0.6
    x_above_idx = variance_list >= 0.6

    x_below = components[np.where(x_below_idx)[0]]
    x_above = components[np.where(x_above_idx)[0]]

    y_below = variance_list[x_below_idx]
    y_above = variance_list[x_above_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_below, y_below, marker='o')
    ax.scatter(x_above, y_above, marker='o', color='red')
    ax.set_xticks(components)
    ax.axhline(0.6, linestyle='--', linewidth=1, color='red')
    ax.set(xlabel="Number of Components",
           ylabel="Variance",
           title="Explained Variance after SVD")


def kmeans_simulation(points, kmeans_class=CustomKMeans, max_centroids=20, skip=5):
    """

    Parameters
    ----------
    points : a numpy of dim (n, d)
    kmeans_class : a class that implements fit_predict() and has attributes n_iter_ and intertia_
                   like the sklearn implementation
    max_centroids : int
        specify how to randomize the initial centroids

    Returns
    -------
    dict {i: {n_iter: int, sse: float, time: float}
        a dictionary providing information about the performance of KMeans
        on different number of centroids
    """
    kmeans_runs = {}

    for k in range(2, max_centroids + 1, skip):
        tic = time.time()

        custom_kmeans = kmeans_class(n_clusters=k)
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker='o')
    ax.set_xticks(x)
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


class ReviewsWordCloud(object):
    def __init__(self, reviews_df, cluster_col='Cluster'):
        self.num_cluster = reviews_df.Cluster.max() + 1
        self.df = reviews_df
        self.cluster_col = cluster_col

    def visualize_wordcloud(self, cluster_num, max_words=50, max_font_size=50, background_color='white'):
        word_freq_dict = cluster_text_as_dict(self.df, cluster_num, cluster_col=self.cluster_col)

        wc = WordCloud(stopwords=STOPWORDS,
                       max_words=max_words,
                       background_color=background_color,
                       max_font_size=max_font_size)

        cloud = wc.generate_from_frequencies(word_freq_dict)

        plt.figure(figsize=(10, 5))
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()