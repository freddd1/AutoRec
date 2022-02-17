from torch.utils.data import Dataset
import pandas as pd


class I_AutoRecDataSet(Dataset):
    """
    This class will hold matrix R represent rating where each row is a item and each col is user. Hence, Item to Item.
          user1 user2
    item1   r     r
    item2   r     r
    """

    def __init__(self, rating: pd.DataFrame):
        super(I_AutoRecDataSet).__init__()
        # we transpose the matrices, so it will be ITEM TO ITEM
        self.rating = rating

        self.rating = self.rating.T.values

        # the mask will indicate us where we have rating. hence, rating > 0
        self.mask = (self.rating > 0)

    def __getitem__(self, idx):
        return self.rating[idx], self.mask[idx]

    def __len__(self):
        return self.rating.shape[0]
