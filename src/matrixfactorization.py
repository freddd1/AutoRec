from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np

# TODO: Create local(maybe global) 'device' variable.


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

    def forward(self, user, item):
        user = self.users_emb(user)
        item = self.items_emb(item)
        return (user*item).sum(axis=1)


def mf_train(train, model, device, batch_size=256, epochs=10, lr=0.001, reg=0):
    # TODO:
    #  1.Create variable that calculates the validation loss during the training.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    dataloader = DataLoader(train.values, batch_size=batch_size)
    best_loss = 5
    for i in range(epochs):
        for batch_id, train_batch in enumerate(dataloader):
            # model.train()
            users = tensor_to_numpy_column(train_batch,0)
            items = tensor_to_numpy_column(train_batch, 1)
            ratings = tensor_to_numpy_column(train_batch, 2)

            users = torch.LongTensor(users).to(device)
            items = torch.LongTensor(items).to(device)
            ratings = torch.FloatTensor(ratings).to(device) # rating is our label

            preds = model(users, items)
            loss = torch.sqrt(nn.functional.mse_loss(preds, ratings))

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # update
            optimizer.step()

            print(f'Train RMSE: {loss.item()}')

            # Save Best Loss score
            if best_loss> loss.data:
                best_loss = loss.data

        print('#####################################')
        print(f'\n Epoch Number: {i+1} \n')
        print(f'\n Currently Best loss : {best_loss} \n')
        print('#####################################')


def tensor_to_numpy_column(batch: DataLoader, column_num: int) -> np.array:
    """
    Helper Function to get the relevant columns (users, items, ratings)
    :param batch: tensor in the shape of Batch_size, holds matrix of (users, items, ratings)
    :param column_num: column index.
    :return: Numpy array with relevant column(users, items, ratings).
    """
    column = list(map(lambda batch: list(map(lambda z: z.tolist() - 1, batch)), batch))
    column = np.array(column)[:, column_num]
    column = list(map(lambda z: torch.tensor(z), column))
    return column



