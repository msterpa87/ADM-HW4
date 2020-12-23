import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from scipy.stats import f_oneway
import re
import pandas as pd
import nltk
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

# constants
tfidf_filename = "tfidf.pkl"
svd_filename = "svd.pkl"
data_filename = "data.pkl"
reviews_pathname = "processed_reviews.csv"
processed_col = "ProcessedText"
no_stem_col = "Text_not_stemmed"
cluster_col = "Cluster"
stopwords = stopwords.words('english')

# user defined words to remove from WordCloud
special_words = ['amazon', 'review', 'something', 'thing', 'buy',
                 'price', 'prices', 'year', 'years','order', 'value',
                 'product', 'years', 'great', 'try', 'purchase',
                 'pay', 'brand', 'way', 'lot', 'stuff', 'time' 'get',
                 'recommend', 'anything', 'buying', 'stores', 'bit',
                 'subscribe', 'size', 'package', 'reviews', 'thing',
                 'things']


def load_data(processed=False):
    """ Data loader of processed_reviews.csv """
    cols = ['ProductId', 'UserId', 'Score', 'Text']
    reviews_df = pd.read_csv(reviews_pathname)

    if processed:
        cols += [processed_col, no_stem_col]

    return reviews_df[cols]


def is_not_noun(tag):
    """ Returns true iff the NLTK tag is a noun """
    if tag[0] == 'N':
        return False
    return True


def words_to_filter(str_list):
    """ Takes a list of reviews and returns a list of stemmed words
        including non-nouns and short words. """
    vocabulary = list(set(words_from_str_list(str_list)))
    tags = nltk.pos_tag(vocabulary)
    to_filter = filter(lambda x: is_not_noun(x[1]) or len(x[0]) < 3, tags)
    to_filter = list(map(lambda x: x[0], to_filter))
    to_filter = to_filter + special_words
    stemmer = SnowballStemmer("english")
    to_filter = [stemmer.stem(w) for w in to_filter]
    return list(set(to_filter))


def preprocess(s, stemming=True):
    """ Takes a string s and runs a standard preprocessing """
    s = s.lower()

    # remove html tags
    s = re.sub(r"<[^>]+>", "", s)

    # remove punctuation
    s = re.sub(r'[^a-zA-Z]', ' ', s)

    # removing digits and special chars
    tokens = list(filter(lambda x: x not in stopwords, s.split()))

    # stemming with SnowballStemmer worked slightly better than Porter and WordNet
    if stemming:
        stemmer = SnowballStemmer("english")
        s = " ".join([stemmer.stem(w) for w in tokens])
    else:
        s = " ".join(tokens)

    return s


def words_from_str_list(str_list):
    """ Shortcut to join a reviews list as a single string """
    return " ".join(str_list).split()


def get_counter(df):
    """ Takes a df with a column ProcessedText of strings and returns
        a Counter object on its corpus"""
    counter = Counter()
    words = words_from_str_list(df[processed_col])
    counter.update(words)
    return counter


def cluster_text_as_dict(reviews_df, cluster_num):
    """ Returns a dictionary of words to be used in the WordCloud representation

    Parameters
    ----------
    reviews_df : dataframe with column 'Cluster' and 'ProcessedText'
    cluster_num : cluster number to return as text

    Returns
    -------
    dict {word: frequency}
    """
    reviews_list = reviews_df[reviews_df[cluster_col] == cluster_num]['Text']
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, max_features=300).fit(reviews_list)
    top_features = tfidf.get_feature_names()
    words = preprocess(" ".join(reviews_list), stemming=False).split()
    tags = nltk.pos_tag(set(list(words)))
    nouns = list(map(lambda x: x[0], filter(lambda x: x[1][0] == 'N', tags)))
    counter = Counter()
    counter.update(words)
    word_freq_dict = {}

    filtered_words = list(set.intersection(set(nouns), set(top_features)))

    for key, val in counter.items():
        if key in filtered_words and key not in special_words:
            word_freq_dict[key] = val

    return word_freq_dict


def product_count(df):
    """ Prints the number of products per cluster

    Parameters
    ----------
    df : a DataFrame with columns 'ProductId' and 'Cluster'
    """
    cols = ['ProductId', cluster_col]
    products_df = df[cols].groupby([cluster_col]).count()
    products_df.rename(columns={'ProductId': 'Number of Products'})
    return products_df


def unique_users(df):
    """ Prints the number of unique users per cluster

    Parameters
    ----------
    df : a DataFrame with columns 'ProductId' and 'Cluster'
    """
    cols = ['UserId', cluster_col]
    users_df = df[cols].groupby([cluster_col]).agg({"UserId": "nunique"})
    users_df.rename(columns={'UserId': 'Unique Users'})
    return users_df



