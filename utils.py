import numpy as np
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from scipy.stats import f_oneway
import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

tfidf_filename = "tfidf.pkl"
svd_filename = "svd.pkl"
data_filename = "data.pkl"
reviews_pathname = 'processed_reviews.csv'
processed_col = 'ProcessedText'
no_stem_col = "Text_not_stemmed"
stopwords = stopwords.words('english')

special_words = ['amazon', 'review', 'something', 'thing', 'buy',
                 'price', 'year', 'order', 'value']


def load_data(processed=False):
    cols = ['ProductId', 'UserId', 'Score', 'Text']
    reviews_df = pd.read_csv(reviews_pathname)

    if processed:
        cols += [processed_col, no_stem_col]

    return reviews_df[cols]


def is_not_noun(tag):
    if tag[0] == 'N':
        return False
    return True


def get_word_freq(df, word):
    freq = len(df[df[no_stem_col].str.contains(word)]) / len(df)
    print("Reviews containing 'product': {:.2%}".format(freq))


def words_to_filter(str_list):
    vocabulary = list(set(words_from_str_list(str_list)))
    tags = nltk.pos_tag(vocabulary)
    to_filter = filter(lambda x: is_not_noun(x[1]) or len(x[0]) < 3, tags)
    to_filter = list(map(lambda x: x[0], to_filter))
    to_filter = to_filter + special_words
    stemmer = SnowballStemmer("english")
    to_filter = [stemmer.stem(w) for w in to_filter]
    return list(set(to_filter))


def preprocess(s, stemming=True):
    s = s.lower()

    # remove html tags
    s = re.sub(r"<[^>]+>", "", s)

    # remove punctuation
    s = re.sub(r'[^a-zA-Z]', ' ', s)

    # removing digits and special chars
    tokens = list(filter(lambda x: x not in stopwords, s.split()))

    if stemming:
        stemmer = SnowballStemmer("english")
        s = " ".join([stemmer.stem(w) for w in tokens])
    else:
        s = " ".join(tokens)

    return s


def words_from_str_list(str_list):
    return " ".join(str_list).split()


def get_counter(df):
    """ Takes a df with a column ProcessedText of strings and returns
        a Counter object on its corpus"""
    counter = Counter()
    words = words_from_str_list(df[processed_col])
    counter.update(words)
    return counter


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


def update_centroids(points, clusters):
    n_centroids = clusters.max()
    return np.array([(points[clusters == i]).mean(axis=0) for i in range(n_centroids + 1)])


def clusters_size(clusters):
    return np.unique(clusters, return_counts=True)[1]


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

