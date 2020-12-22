from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import PorterStemmer
from scipy.stats import f_oneway
import re
import string
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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
    def __init__(self, max_features=100, n_components=20, min_df=0.2, save=True):
        self.save = save
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.tfidf = None
        self.svd = None
        self.data = None

    def fit_transform(self, reviews):
        """
        Returns a matrix of tfidf with reduced dimensions
        :param reviews: a list of strings representing a user review
        :return: a sparse matrix, each row represents the tfidf of the input text
        """
        tfidf = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            min_df=self.min_df
        )

        reviews_tfidf = tfidf.fit_transform(reviews)

        # guarantees that n_components < n_features
        n_components = min(self.n_components, reviews_tfidf.shape[1] - 1)

        # dimensionality reduction
        svd = TruncatedSVD(n_components=n_components)
        data = svd.fit_transform(reviews_tfidf)

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
    return ((points[:, None, :] - centroids[None, :, :])**2).sum(axis=2).argmin(axis=1)


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


def compute_distortion(points, centroids):
    return distance(points, centroids).mean()


def update_centroids(points, clusters):
    n_centroids = clusters.max()
    return np.array([(points[clusters == i]).mean(axis=0) for i in range(n_centroids + 1)])


def clusters_size(clusters):
    return np.unique(clusters, return_counts=True)[1]


def sharding_init(data, n_centroids):
    """ Returns k centroids from data using the naive sharding
        https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html """
    sharding_idxs = data.sum(axis=1).argsort()
    chunks = np.array_split(data[sharding_idxs], n_centroids, axis=0)
    centroids = [chunks[i].mean(axis=0) for i in range(n_centroids)]

    return np.array(centroids)


def random_init(points, n_clusters):
    """ Assign centroids by randomly sampling among points """
    random_indices = np.random.choice(range(len(points)), n_clusters, replace=False)

    return points[random_indices]


def cluster_as_text(reviews_df, cluster_num, cluster_col='Cluster', text_col='Text'):
    """

    :param cluster_col:
    :param text_col:
    :param reviews_df: dataframe with column 'Cluster' and 'ProcessedText'
    :param cluster_num: cluster number to return as text
    :return: string
    """
    return " ".join(reviews_df[reviews_df[cluster_col] == cluster_num][text_col].astype(str).tolist())


def product_count(df):
    cols = ['ProductId', 'Cluster']
    products_df = df[cols].groupby(['Cluster']).count()
    products_df.rename(columns={'ProductId': 'Number of Products'})
    return products_df


def unique_users(df):
    cols = ['UserId', 'Cluster']
    users_df = df[cols].groupby(['Cluster']).agg({"UserId": "nunique"})
    users_df.rename(columns={'UserId': 'Unique Users'})
    return users_df


def anova(df):
    n = df['Cluster'].max() + 1
    groups = [df[df['Cluster'] == i]['Score'].values for i in range(n)]
    normalized_groups = list(map(lambda x: (x - x.mean()) / x.std(), groups))
    f, p = f_oneway(*normalized_groups)
    

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

