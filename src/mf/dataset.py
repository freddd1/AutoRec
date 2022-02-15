import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MFDataSet(Dataset):
    """
    This class will hold data for MF algorithm
    The data will be with ['user_id', 'items_id', 'rating'] columns
    """
    def __init__(self, rating: pd.DataFrame):
        """
        :param rating: pd.DataFrame with ['user_id', 'items_id', 'rating'] columns.
                       It is ok if we have more columns, but it must include ['user_id', 'items_id', 'rating']
        """
        super(MFDataSet).__init__()

        # NOTE: that we are changing the indexes to start from 0.
        # Therefore, to get the "original id" we should add 1 to it
        self.users_ids = rating.user_id.values
        self.items_ids = rating.item_id.values

        self.ratings = rating.rating.values

    def __getitem__(self, idx):
        return self.users_ids[idx], self.items_ids[idx], self.ratings[idx]

    def __len__(self):
        return len(self.ratings)

