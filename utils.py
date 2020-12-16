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


reviews_pathname = 'reviews/Reviews.csv'
processed_reviews_pathname = 'reviews/processed_reviews.csv'


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
    def __init__(self, min_df=0.1, max_df=0.9, max_features=100, n_components=20, save=True):
        self.save = save
        self.n_components = n_components
        self.max_df = max_df
        self.max_features = max_features
        self.min_df = min_df
        self.variance_ = None

    def reviews_to_vectors(self, reviews):
        """
        Returns a matrix of tfidf with reduced dimensions
        :param reviews: a list of strings representing a user review
        :return: a sparse matrix, each row represents the tfidf of the input text
        """
        tfidf = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
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
            pickle.dump(data, open("tfidf_matrix.pkl", "wb"))

        self.variance_ = svd.explained_variance_

        return data


def distance(point, centroids):
    """

    :param centroids: vector of dimensione (1,d)
    :param point: vector of dimension (k,d)
    :return: vector of floats of dimension d
    """
    return ((point - centroids) ** 2).sum(axis=1)


def get_closest(point, centroids):
    """

    :param point: vector of dim (1,d)
    :param centroids: vector of dim (k,d)
    :return: index of the centroid closest to point
    """
    return distance(point, centroids).argmin()


def average(points):
    """

    :param points:
    :return:
    """
    return np.average(points, axis=0)


def compute_sse(points, centroids, assigned_centroids):
    """

    :param assigned_centroids:
    :param points:
    :param centroids:
    :return:
    """
    return distance(points, centroids[assigned_centroids]).sum() / len(points)


def update_centroids(points, clusters, n_centroids):
    """

    :param n_centroids:
    :param points:
    :param clusters:
    :return:
    """
    return np.array([average(points[clusters == i]) for i in range(n_centroids)])


def clusters_size(clusters):
    return np.unique(clusters, return_counts=True)[1]


class CustomKMeans(object):
    def __init__(self, n_centroids=10, max_iter=100):
        self.n_centroids = n_centroids
        self.max_iter = max_iter
        self.sse_list_ = []
        self.total_time_ = None
        self.iter_time_ = None

    def predict(self, points):
        tic = time.time()

        # cluster assignment
        n_points = len(points)
        assigned_centroids = np.zeros(n_points, dtype=int)
        n_centroids = self.n_centroids
        max_iter = self.max_iter

        # initialize random centroids
        random_indices = np.random.choice(range(n_points), n_centroids, replace=False)
        centroids = points[random_indices]

        for i in range(max_iter):

            # get cluster assignement for each point
            assigned_centroids = np.array([get_closest(point, centroids) for point in points])

            # update centroids taking average point
            centroids = update_centroids(points, assigned_centroids, n_centroids)

            # computer squared error
            self.sse_list_.append(compute_sse(points, centroids, assigned_centroids))

        toc = time.time()

        total = toc - tic
        self.total_time_ = total
        self.iter_time_ = total / max_iter
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


def load_reviews(filename=reviews_pathname, clusters=None):
    cols = ['ProductId', 'UserId', 'Score', 'Text']
    reviews_df = pd.read_csv(reviews_pathname, usecols=cols)
    reviews_df['Text'] = reviews_df['Text'].apply(lambda x: preprocess(x, stemming=False))

    if clusters is not None:
        reviews_df['Cluster'] = clusters

    return reviews_df


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
        assert(cluster_num < self.num_cluster)

        wordcloud = WordCloud(stopwords=STOPWORDS,
                              max_words=max_words,
                              background_color=background_color,
                              max_font_size=max_font_size).generate(self.texts[cluster_num])

        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
