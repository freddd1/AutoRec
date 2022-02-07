import torch
from torch import nn
from torch.utils.data import DataLoader
# TODO: Create local(maybe global) 'device' variable.


class AutoRec(nn.Module):
    """
    AutoRec model. See explanation on :param: num_features for use as USER TO USER or ITEM TO ITEM
    """
    def __init__(self, num_features, num_hidden=500):
        """
        :param num_hidden: Size of the hidden layer
        :param num_features: If num_features == num_items that means that we are doing USER TO USER model.
                             If num_features == num_users that means that we are doing ITEM TO ITEM model.
                             The logic is the a user vector has number of items features for it.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_hidden, num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def autorec_train(train, model, device, batch_size=64, epochs=10, lr=0.005, reg=0):
    # TODO:
    #  1.Create variable that calculates the validation loss during the training.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    dataloader = DataLoader(train.values, batch_size=batch_size)
    for epoch in range(1, epochs+1):
        for batch_id, train_batch in enumerate(dataloader):
            # model.train()
            train_batch = train_batch.float().to(device)
            preds = model(train_batch).to(device)

            loss = torch.sqrt(nn.functional.mse_loss(preds, train_batch))
            # print(f'epoch: {epoch}, batch: {batch_id}, loss: {loss.item()}')

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # update
            optimizer.step()
        print(f'epoch: {epoch} train RMSE: {loss.item()}')

    return preds

