from scipy.stats import ttest_ind
from kmeans import *
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud, STOPWORDS
from utils import cluster_text_as_dict
import seaborn as sns
sns.set_theme(style="white")
plt.style.use("seaborn-whitegrid")


def plot_variance(tfidf, skip=1, save_data=True):
    """ Plots the variance by number of components in SVD """
    n = tfidf.shape[1]
    variance_list = []
    components = []

    # runs multiple instance of TruncatedSVD with varying number of components
    for i in range(2, n, skip):
        svd = TruncatedSVD(n_components=i)
        svd.fit_transform(tfidf)

        # saves n_components and variance for plotting
        variance = svd.explained_variance_.sum()
        variance_list.append(variance)
        components.append(i)

    components = np.array(components)
    variance_list = np.array(variance_list)

    # saves data for convenience
    if save_data:
        with open("variance_data.pkl", "wb") as f:
            pickle.dump([components, variance_list], f)

    # plots points below and above threshold with different colors
    x_below_idx = variance_list < 0.6
    x_above_idx = variance_list >= 0.6
    x_below = components[np.where(x_below_idx)[0]]
    x_above = components[np.where(x_above_idx)[0]]
    y_below = variance_list[x_below_idx]
    y_above = variance_list[x_above_idx]

    # plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_below, y_below, marker='o')
    ax.scatter(x_above, y_above, marker='o', color='red')
    ax.set_xticks(components)
    ax.axhline(0.6, linestyle='--', linewidth=1, color='red')
    ax.set(xlabel="Number of Components",
           ylabel="Variance",
           title="Explained Variance after SVD")


def kmeans_simulation(points, kmeans_class=CustomKMeans, max_centroids=20, skip=5, save_data=True):
    """ Executes multiple runs of kmeans_class (either CustomKMeans or sklearn.cluster.KMeans)

    Parameters
    ----------
    points : a numpy of dim (n, d)
    kmeans_class : a class that implements fit() and has attributes n_iter_ and inertia_
                   like the sklearn implementation
    max_centroids : int
        maximum number of centroids to try
    skip : int
        step_size of number of clusters to run KMeans

    Returns
    -------
    dict {i: {n_iter: int, sse: float, time: float}
        a dictionary providing information about the performance of KMeans
        on different number of centroids
    """
    kmeans_runs = {}

    # run KMeans with different number of clusters
    for k in range(2, max_centroids + 1, skip):
        tic = time.time()

        custom_kmeans = kmeans_class(n_clusters=k)
        custom_kmeans.fit(points)

        toc = time.time()

        sse = int(custom_kmeans.inertia_)
        n_iter = custom_kmeans.n_iter_
        time_ = round(toc - tic, 2)

        kmeans_runs[k] = dict(n_iter=n_iter, sse=sse, time=time_)

    # save data for convenience
    if save_data:
        with open("simulation_data.pkl", "wb") as f:
            pickle.dump(kmeans_runs, f)

    return kmeans_runs


def plot_sse(data):
    """ Plots the sum of squares error by number of clusters

    Parameters
    ----------
    data : dictionary returned from kmeans_simulation()

    """
    x = list(data.keys())
    y = np.array([d['sse'] for d in data.values()])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker='o')
    ax.set_xticks(x)
    plt.title("SSE by Number of Clusters", fontsize=15)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squares Error")


def plot_sse_comparison(sklearn_data, custom_data):
    """ Plots a comparison of SSE between CustomKMeans and sklearn

    Parameters
    ----------
    sklearn_data and custom_data : dictionaries returned from kmeans_simulation()

    """
    sklearn_df = pd.DataFrame.from_dict(sklearn_data, orient='index')
    custom_df = pd.DataFrame.from_dict(custom_data, orient='index')
    x = list(sklearn_df.index)
    plt.xticks(x)
    plt.plot(x, sklearn_df["sse"], marker='o')
    plt.plot(x, custom_df["sse"], marker='o')
    plt.legend(["sklearn kmeans", "custom kmeans"])
    plt.title("Sklearn vs Custom KMeans SSE", fontsize=15)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squares Error")


def plot_performance(sklearn_data, custom_data):
    """ Same asa plot_sse_comparison() but compares execution times """
    sklearn_df = pd.DataFrame.from_dict(sklearn_data, orient='index')
    custom_df = pd.DataFrame.from_dict(custom_data, orient='index')
    x = list(sklearn_df.index)
    plt.xticks(x)
    plt.plot(x, sklearn_df["time"], marker='o')
    plt.plot(x, custom_df["time"], marker='o')
    plt.legend(["sklearn kmeans", "custom kmeans"])
    plt.title("Sklearn vs Custom KMeans (Time)", fontsize=15)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Time (s)")


class ReviewsWordCloud(object):
    def __init__(self, reviews_df):
        self.df = reviews_df

    def visualize_wordcloud(self, cluster_num, max_words=50, max_font_size=50, background_color='white'):
        """ Visualize a WordCloud of the specified cluster

        Parameters
        ----------
        cluster_num : int - cluster to visualize
        max_words : int - maximum number of words in the cloud
        max_font_size : int - maximum font size
        background_color : str - cloud background

        """
        word_freq_dict = cluster_text_as_dict(self.df, cluster_num)

        wc = WordCloud(stopwords=STOPWORDS,
                       max_words=max_words,
                       background_color=background_color,
                       max_font_size=max_font_size)

        cloud = wc.generate_from_frequencies(word_freq_dict)

        plt.figure(figsize=(10, 5))
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()


def plot_stats(df):
    """ Plots the matrix of p-values resulting from a sample t-test
        of mean differences among each pair of clusters """
    n = df['Cluster'].max() + 1
    test_matrix = np.zeros((n, n))
    distr = [df[df['Cluster'] == i]['Score'] for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            _, p_value = ttest_ind(distr[i], distr[j])
            test_matrix[i, j] = p_value

    mask = np.tril(np.ones_like(test_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.set(rc={'figure.figsize': (10, 5)})
    sns.set_theme(style="white")
    ax = sns.heatmap(test_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title("T-test p-value Matrix", fontsize=15)