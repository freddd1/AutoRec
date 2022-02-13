from torch import nn


class MatrixFactorization(nn.Module):
    """
    matrix factorization model.
    """
    # TODO:
    #  Add bias to the model
    def __init__(self, num_users, num_items, k=15):
        """
        :param num_users: number of unique users
        :param num_items: number of unique items
        :param k: k is the size of the latent space
        """
        super().__init__()
        self.k = k
        self.num_users = num_users
        self.num_items = num_items

        # create the latent matrixs
        self.users_emb = nn.Embedding(num_users, k)
        self.items_emb = nn.Embedding(num_items, k)

        # # Init weights
        # self.users_emb.weight.data.uniform_(0, 0.05)
        # self.items_emb.weight.data.uniform_(0, 0.05)

    def forward(self, user, item):
        user = self.users_emb(user)
        item = self.items_emb(item)
        return (user*item).sum(axis=1)

    def params(self):
        return {'k': self.k}