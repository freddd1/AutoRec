from torch import nn
import torch

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


def mf_train(train, model, device, epochs=10, lr=0.001, reg=0):
    # TODO:
    #  1.Create variable that calculates the validation loss during the training.
    #  2.Modify the function to use batch size.(after dataloader class ready)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    for i in range(epochs):
        model.train()
        # users and items indexes start 1, therefore we use the -1
        users = torch.LongTensor(train.user_id.values-1).to(device)
        items = torch.LongTensor(train.item_id.values-1).to(device)
        ratings = torch.FloatTensor(train.rating.values).to(device) # rating is our label

        preds = model(users, items)
        loss = torch.sqrt(nn.functional.mse_loss(preds, ratings))

        # backpropagation
        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()

        print(f'train RMSE: {loss.item()}')


#Data loader
def mf_dataloader():
    '''
    '''
    # TODO:
    #  Build data loader function that returns dataloader class from pytorch. just like in autorec class.
    pass

