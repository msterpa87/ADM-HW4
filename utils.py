import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re


def preprocess(s):
    """
    Preprocess input string to be used in BOW representation
    :param s: string
    :return: string
    """
    s = s.lower()

    # expand contracted verbs
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"won\'t", "will not", s)
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ve'", " have", s)
    s = re.sub(r"\'m", " am", s)

    # remove html tags
    s = re.sub(r"<[^>]+>", "", s)

    # removing punctuations
    exclist = string.punctuation + string.digits
    table_ = str.maketrans(exclist, ' ' * len(exclist))
    s = " ".join(stem(w) for w in s.translate(table_).split())

    return s