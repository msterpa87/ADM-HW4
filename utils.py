from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
import pickle
import matplotlib.pyplot as plt
import time

tfidf_filename = "tfidf.pkl"
svd_filename = "svd.pkl"
data_filename = "data.pkl"


def preprocess(s, decontract=True, remove_html=True, punctuation=True, stemming=True):
    """

    :param s:
    :param decontract:
    :param remove_html:
    :param punctuation:
    :param stemming:
    :return:
    """
    s = s.lower()

    if decontract:
        # expand contracted verbs
        s = re.sub(r"\'s", " is", s)
        s = re.sub(r"\'ll", " will", s)
        s = re.sub(r"won\'t", "will not", s)
        s = re.sub(r"n\'t", " not", s)
        s = re.sub(r"\'d", " would", s)
        s = re.sub(r"\'ve'", " have", s)
        s = re.sub(r"\'m", " am", s)

    if remove_html:
        # remove html tags
        s = re.sub(r"<[^>]+>", "", s)

    if punctuation:
        # removing punctuations and stemming
        exclist = string.punctuation + string.digits
        table_ = str.maketrans(exclist, ' ' * len(exclist))
        s = " ".join(s.translate(table_).split())

    if stemming:
        stemmer = PorterStemmer()
        s = " ".join(stemmer.stem(w) for w in s.split())

    return s


class Vectorizer(object):
    def __init__(self, max_features=100, n_components=20, save=True):
        self.save = save
        self.n_components = n_components
        self.max_features = max_features
        self.tfidf = None
        self.svd = None
        self.data = None

    def load_from_file(self):
        self.tfidf = pickle.load(open(tfidf_filename, "rb"))
        self.svd = pickle.load(open(svd_filename, "rb"))
        self.data = pickle.load(open(data_filename, "rb"))

    def save_to_file(self):
        pickle.dump(self.tfidf, open(tfidf_filename, "wb"))
        pickle.dump(self.svd, open(svd_filename, "wb"))
        pickle.dump(self.data, open(data_filename, "wb"))

    def fit_transform(self, reviews):
        """
        Returns a matrix of tfidf with reduced dimensions
        :param reviews: a list of strings representing a user review
        :return: a sparse matrix, each row represents the tfidf of the input text
        """
        tfidf = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english'
        )

        reviews_tfidf = tfidf.fit_transform(reviews)

        # gurantees that n_components < n_features
        n_components = min(self.n_components, reviews_tfidf.shape[1] - 1)

        # dimensionality reduction
        svd = TruncatedSVD(n_components=n_components)
        data = svd.fit_transform(reviews_tfidf)

        if self.save:
            # saving pickles for future loading
            self.svd = svd
            self.tfidf = tfidf
            self.data = data
            self.save_to_file()

        return data

    def get_feature_names(self):
        return self.tfidf.get_feature_names()

    def get_components(self):
        return self.svd.components_

    def get_variance(self):
        return self.svd.explained_variance_


def distance(points, centroids):
    """ Returns the vector of distances of points to respective centroids

    Parameters
    ----------
    points : numpy (n, 1, d)
    centroids : numpy (1, k, d)
        centroids[i] = closest centroid to point[i]
    Returns
    -------
    distances : numpy (n, d)
        the vector of distance of each point from its centroid
    """
    return ((points - centroids) ** 2).sum(axis=points.ndim - 1)


def get_closest(point, centroids):
    """

    :param point: vector of dim (n, 1, d)
    :param centroids: vector of dim (1, k,d)
    :return: indices of the centroids closest to each point
    """
    return distance(point, centroids).argmin(axis=1)


def average(points):
    """ Average point of the given set of points """
    return np.average(points, axis=0)


def compute_sse(points, centroids):
    return distance(points, centroids).sum()


def update_centroids(points, clusters, n_centroids):
    return np.array([(points[clusters == i]).mean(axis=0) for i in range(n_centroids)])


def clusters_size(clusters):
    return np.unique(clusters, return_counts=True)[1]


def sharding_init(data, n_centroids):
    """ Returns k centroids from data using the naive sharding
        https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html

    Parameters
    ----------
    data : numpy (n, d)
    n_centroids : int

    Returns
    -------
    centroids
        numpy (k, d)
    """
    sharding_idxs = data.sum(axis=1).argsort()
    chunks = np.array_split(data[sharding_idxs], n_centroids, axis=0)
    centroids = [chunks[i].mean(axis=0) for i in range(n_centroids)]

    return np.array(centroids)


def random_init(points, n_centroids):
    random_indices = np.random.choice(range(len(points)), n_centroids, replace=False)

    return points[random_indices]


class CustomKMeans(object):
    def __init__(self, n_centroids=10, max_iter=100, init='random', threshold_pct=0.001):
        """
        Parameters
        ----------
        n_centroids : int
        max_iter : int
        init : ['random', 'sharding']
        """
        self.n_centroids = n_centroids
        self.max_iter = max_iter
        self.n_iter_ = 0
        self.inertia_ = []  # will contain the sum of squares error of each iteration
        self.threshold_pct = threshold_pct

        if init == 'random':
            self.init = random_init
        elif init == 'sharding':
            self.init = sharding_init

    def predict(self, points):
        tic = time.time()

        # cluster assignment
        n_points = len(points)
        assigned_centroids = np.zeros(n_points, dtype=int)
        n_centroids = self.n_centroids
        max_iter = self.max_iter

        # initialize centroids
        centroids = self.init(points, n_centroids)

        last_sse = 0

        for i in range(max_iter):
            # get cluster assignement for each point
            assigned_centroids = get_closest(points[:, None, :], centroids[None, :, :])

            # update centroids taking average point
            centroids = update_centroids(points, assigned_centroids, n_centroids)

            # computer squared error
            current_sse = int(compute_sse(points, centroids[assigned_centroids]))
            if (abs(last_sse - current_sse) / current_sse) < self.threshold_pct:
                self.n_iter_ = i + 1
                break
            last_sse = current_sse

        self.inertia_ = compute_sse(points, centroids[assigned_centroids])

        return assigned_centroids


def cluster_as_text(reviews_df, cluster_num, cluster_col='Cluster', text_col='Text'):
    """

    :param cluster_col:
    :param text_col:
    :param reviews_df: dataframe with column 'Cluster' and 'ProcessedText'
    :param cluster_num: cluster number to return as text
    :return: string
    """
    return " ".join(reviews_df[reviews_df[cluster_col] == cluster_num][text_col].astype(str).tolist())


class ReviewsWordCloud(object):
    def __init__(self, reviews_df, text_col='Text', cluster_col='Cluster'):
        self.num_cluster = reviews_df.Cluster.max() + 1

        # saves the text content of each cluster
        self.texts = [cluster_as_text(reviews_df, i, cluster_col=cluster_col, text_col=text_col)
                      for i in range(self.num_cluster)]

    def visualize_wordcloud(self, cluster_num, max_words=50, max_font_size=50, background_color='white'):
        """

        :param max_font_size:
        :param background_color:
        :param max_words:
        :param cluster_num:
        :return:
        """
        assert (cluster_num < self.num_cluster)

        wordcloud = WordCloud(stopwords=STOPWORDS,
                              max_words=max_words,
                              background_color=background_color,
                              max_font_size=max_font_size).generate(self.texts[cluster_num])

        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

