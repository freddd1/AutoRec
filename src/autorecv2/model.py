from torch import nn
class AutoRecV2(nn.Module):
    """
    AutoRec model.
    """
    def __init__(self, num_features, num_hidden=500, dropout=0):
        """
        :param num_hidden: Size of the hidden layer
        """
        super().__init__()

        self.num_hidden = num_hidden
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.Dropout(self.dropout),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_hidden, num_features),
            nn.Identity()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def params(self):
        return {'num_hidden': self.num_hidden, 'dropout': self.dropout}