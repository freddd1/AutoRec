import pandas as pd
from sklearn.model_selection import train_test_split

MOVIELENS_DATA_PATH = 'data/movielens/ml-100k/'
NETFLIX_DATA_PATH = 'data/netflix'


def movielens_load(fold: int = None) -> (pd.DataFrame, pd.DataFrame):
    """
    The function will return test and train (or validation) datasets.
    :param fold: in range[1,5] that represents fold. If fold == None (Default), it will return the full dataset.
    :return: (train, test), each with ['user_id', 'items_id', 'rating', 'timestamp'] cols
    """
    cols_data = ['user_id', 'item_id', 'rating', 'timestamp']

    if fold:
        files_path = f'{MOVIELENS_DATA_PATH}u{fold}'
    else:
        files_path = f'{MOVIELENS_DATA_PATH}ua'

    train = pd.read_csv(f'{files_path}.base', delimiter='\t', header=None, names=cols_data)
    test = pd.read_csv(f'{files_path}.test', delimiter='\t', header=None, names=cols_data)

    return train, test


def movielens_create_ratings(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    the function will convert the raw train and test to dataframes for rating
    where each row is user_id and each column is item_id.
    :param train: raw train from movielens
    :param test: raw test from movielens
    :return: (train, test) each as matrix r with size(num_users, num_items)
    """
    train = train.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    test = test.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    return train, test


def movielens_prep(fold: int = None) -> (pd.DataFrame, pd.DataFrame):
    """
    this will be the driver function of the movielens data preparation. It will call all other necessary functions.
    :param fold: in range[1,5] that represents fold. If fold == None (Default), it will return the full dataset.
    :return: (train, test)
    """
    train, test = movielens_load(fold=fold)
    train, test = movielens_create_ratings(train, test)
    return train, test

# TODO:
#  create dataloder for the netflix datasets
#  create train and test.
#  create rating matrix

def netflix_load(fold:int = 1) -> (pd.DataFrame):
    """
    The function will convert txt file of the dataset to pd.DataFrame
    :param fold: in the range[1,4]
    :return: df
    """
    cols_data = ['user_id', 'item_id', 'rating', 'timestamp']
    users_id, ratings, timesteps, items_id = [], [], [], []

    file_path = f'data/netflix/combined_data_{fold}.txt'

    with open(file_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()  # strip unwanted spaces

        # saves the movie id
        if line.endswith(':'):
            movie_id = line[:-1]
        else:
            line = line.split(',')
            users_id.append(line[0])
            ratings.append(line[1])
            timesteps.append(line[2])
            items_id.append(movie_id)

    df = pd.DataFrame(list(zip(users_id, items_id, ratings, timesteps)), columns=cols_data)
    df['user_id'] = df.user_id.astype(int)
    df['item_id'] = df.item_id.astype(int)
    df['rating'] = df.rating.astype('int8')
    return df


def netflix_create_ratings(df:pd.DataFrame) -> pd.DataFrame:
    """
    the function will convert the raw train and test to dataframes for rating
    where each row is user_id and each column is item_id.
    :param df:
    :return: rating matrix
    """
    df = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    return df


def netflix_test_train_split(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """
    the function will convert Netflix rating matrix into 2 Dataframes- train & test.
    :param
        df: Netflix rating matrix
        test_size:

    :return:  (train(pd.DataFrame), test(pd.DataFrame))
    """
    train, test = train_test_split(df, test_size=test_size, random_state=17)
    return train, test

