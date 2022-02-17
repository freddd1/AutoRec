import pandas as pd
from sklearn.model_selection import train_test_split

MOVIELENS_100K_PATH = 'data/movielens/ml-100k/'
MOVIELENS_1M_PATH = 'data/movielens/ml-1m/'


def movielens_load_1m(seed: int = 14, test_size: float = 0.1) -> (pd.DataFrame, pd.DataFrame):
    """
    The function will return test and train (or validation) datasets.
    :param seed: randon state seed.
    :param test_size: the test size.
    :return: (train, test), each with ['user_id', 'items_id', 'rating', 'timestamp'] cols
    """
    cols_data = ['user_id', 'item_id', 'rating', 'timestamp']

    file_path = MOVIELENS_1M_PATH + 'ratings'
    df = pd.read_csv(f'{file_path}.dat', delimiter="::", header=None, names=cols_data)

    # set the user_id and item_id to start from 0
    df.user_id = df.user_id - 1
    df.item_id = df.item_id - 1

    train, test = train_test_split(df, random_state=seed, test_size=test_size)

    return train, test


def movielens_create_ratings_1m(size: tuple = (6040, 3952)) -> (pd.DataFrame, pd.DataFrame):
    """
    the function will convert the raw train and test to dataframes for rating
    where each row is user_id and each column is item_id.
    :param size: Will be the size of the full r matrix = (num_users, num_items)
                 In case of "ml-1m" it is (6040, 3952).
                 You can find the full size in the README of each dataset.
    :return: (train, test), each as matrix r with size(num_users, num_items)
    """
    # load the dataset
    train, test = movielens_load_1m()

    train = train.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    test = test.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    return _helper_createfulldf(train, test, size=size)


def movielens_load_100k(fold: int = None) -> (pd.DataFrame, pd.DataFrame):
    """
    The function will return test and train (or validation) datasets.
    :param fold: in range[1,5] that represents fold. If fold == None (Default), it will return the full dataset.
    :return: (train, test), each with ['user_id', 'items_id', 'rating', 'timestamp'] cols
    """
    cols_data = ['user_id', 'item_id', 'rating', 'timestamp']

    if fold:
        files_path = f'{MOVIELENS_100K_PATH}u{fold}'
    else:
        files_path = f'{MOVIELENS_100K_PATH}ua'

    train = pd.read_csv(f'{files_path}.base', delimiter='\t', header=None, names=cols_data)
    test = pd.read_csv(f'{files_path}.test', delimiter='\t', header=None, names=cols_data)

    for df in [train, test]:
        df.user_id = df.user_id - 1
        df.item_id = df.item_id - 1

    return train, test


def movielens_create_ratings_100k(fold: int = None, size: tuple = (943, 1682)) -> (pd.DataFrame, pd.DataFrame):
    """
    the function will convert the raw train and test to dataframes for rating
    where each row is user_id and each column is item_id.
    :param size: Will be the size of the full r matrix = (num_users, num_items)
                 In case of "ml-110k" it is (943, 1682).
                 You can find the full size in the README of each dataset.
    :param fold: in range[1,5] that represents fold. If fold == None (Default), it will return the full dataset.
    :return: (train, test), each as matrix r with size(num_users, num_items)
    """
    # load the dataset
    train, test = movielens_load_100k(fold=fold)

    train = train.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    test = test.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    return _helper_createfulldf(train, test, size=size)


def _helper_createfulldf(train: pd.DataFrame, test: pd.DataFrame, size) -> (pd.DataFrame, pd.DataFrame):
    """
    the function will create the full rating matrix in the size of the dataset
    so the train and test will be ecxtly the same size and fill with 0 where there is no values.
    :param train: users_items metrix with users and items from the fold
    :param test: users_items metrix with users and items from the fold
    :return: (train, test) each as matrix r with size(num_users, num_items)
    """
    new_train = pd.DataFrame(0, index=range(size[0]), columns=range(size[1]))
    new_test = new_train.copy()

    new_train.loc[train.index, train.columns] = train.values
    new_test.loc[test.index, test.columns] = test.values

    assert new_train.shape == new_test.shape

    return new_train, new_test
