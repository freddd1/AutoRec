import pandas as pd
from sklearn.model_selection import train_test_split

MOVIELENS_DATA_PATH = 'data/movielens/ml-100k/'
NETFLIX_DATA_PATH = 'data/netflix'


MOVIELENS_DATA_PATH = 'data/movielens/ml-100k/'

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

    for df in [train, test]:
        df.user_id = df.user_id - 1
        df.item_id = df.item_id - 1

    return train, test


def movielens_create_ratings(fold: int = None, size: tuple = (943, 1682)) -> (pd.DataFrame, pd.DataFrame):
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
    train, test = movielens_load(fold=fold)

    train = train.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    test = test.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    return _helper_createfulldf(train, test, size=size)


def _helper_createfulldf(train: pd.DataFrame, test:pd.DataFrame, size) -> (pd.DataFrame, pd.DataFrame):
    """
    the function will create the full rating matrix in the size of the dataset
    so the train and test will be ecxtly the same size and fill with 0 where there is no values.
    :param train: users_items metrix with users and items from the fold
    :param test: users_items metrix with users and items from the fold
    :return: (train, test) each as matrix r with size(num_users, num_items)
    """
    new_train = pd.DataFrame(0, index=range(size[0]), columns=range(size[1]))
    new_test = train.copy()

    new_train.loc[train.index, train.columns] = train.values
    new_test.loc[test.index, test.columns] = test.values

    assert new_train.shape == new_test.shape

    return new_train, new_test

# TODO:
#  create dataloder for the netflix datasets
#  create train and test.
#  create rating matrix

def _netflix_load_helper(fold:int = 1) -> (pd.DataFrame):
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

def netflix_load(fold:int = 1) -> (pd.DataFrame, pd.DataFrame, dict, dict):
    df = _netflix_load_helper(fold=fold)
    df, idx2userid, idx2itemid = _arrange_indexes(df)
    train, test = train_test_split(df, test_size=0.15, random_state=14)
    return train, test, idx2userid, idx2itemid

def _arrange_indexes(df:pd.DataFrame) -> (pd.DataFrame, dict, dict):
    """
    :param df: rating df with (atleast) ['item_id', 'user_id', 'rating'] columns
    :return: the new df and idx2userid dict which help us return to the original id and idx2itemid which will do the same, but for the items
    """
    # It is possible that we are missing few users_ids and items_ids, therefore we need to
    # arrange them in a way that our model will work. Arrange them to start from 0
    # and increment by 1.

    # Users
    unique_users_ids = np.unique(df.user_id)
    userid2idx = {old_id: id for id, old_id in enumerate(unique_users_ids)}
    # idx2userid dict will help us go back from the new id to the old one.
    idx2userid = {id: old_id for old_id, id  in userid2idx.items()}

    df.user_id = df.user_id.apply(lambda x: userid2idx[x])

    # Items
    unique_items_ids = np.unique(df.item_id)
    itemid2idx = {old_id: id for id, old_id in enumerate(unique_items_ids)}
    # idx2itemid dict will help us go back from the new id to the old one.
    idx2itemid = {id: old_id for old_id, id  in userid2idx.items()}

    df.item_id = df.item_id.apply(lambda x: itemid2idx[x])

    return df, idx2userid, idx2itemid

def netflix_create_ratings(fold: int=1) -> (pd.DataFrame, pd.DataFrame):
    """
    the function will convert the raw train and test to dataframes for rating
    where each row is user_id and each column is item_id.
    :param fold: in range[1,4] that represents fold.
    :return: (train, test), each as matrix r with size(num_users, num_items)
    """
    # load the dataset
    train, test, idx2userid, idx2itemid = netflix_load(fold=fold)

    train = train.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    test = test.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    size = (np.unique(train.user_id), np.unique(train.item_id))

    train, test = _helper_createfulldf(train, test, size=size)

    return train, test, idx2userid, idx2itemid


