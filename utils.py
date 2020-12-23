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

tfidf_filename = "tfidf.pkl"
svd_filename = "svd.pkl"
data_filename = "data.pkl"
reviews_pathname = 'processed_reviews.csv'
processed_col = 'ProcessedText'
no_stem_col = "Text_not_stemmed"
stopwords = stopwords.words('english')

special_words = ['amazon', 'review', 'something', 'thing', 'buy',
                 'price', 'prices', 'year', 'years','order', 'value',
                 'product', 'years', 'great', 'try', 'purchase',
                 'pay', 'brand', 'way', 'lot', 'stuff', 'time' 'get',
                 'recommend', 'anything', 'buying', 'stores', 'bit',
                 'subscribe', 'size', 'package', 'reviews', 'thing',
                 'things']


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


def cluster_text_as_dict(reviews_df, cluster_num, cluster_col='Cluster', text_col=no_stem_col):
    """

    :param cluster_col:
    :param text_col:
    :param reviews_df: dataframe with column 'Cluster' and 'ProcessedText'
    :param cluster_num: cluster number to return as text
    :return: string
    """
    reviews_list = reviews_df[reviews_df['Cluster'] == cluster_num]['Text']
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

