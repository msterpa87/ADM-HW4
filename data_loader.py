import pandas as pd
from utils import preprocess

reviews_pathname = 'Reviews.csv'
processed_col = 'ProcessedText'


def load_data(processed=False):
    """ Loads the reviews.csv dataset

    Parameters
    ----------
    processed : bool
        if True the output also has a 'ProcessedText' column
    Returns
    -------
        pandas DataFrame
    """
    cols = ['ProductId', 'UserId', 'Score', 'Text']
    reviews_df = pd.read_csv(reviews_pathname)

    if processed:
        cols.append(processed_col)

        if processed_col not in reviews_df.columns:
            reviews_df[processed_col] = reviews_df['Text'].apply(preprocess).astype('U')

            # save for future loading
            reviews_df.to_csv(reviews_pathname)

    return reviews_df[cols]



