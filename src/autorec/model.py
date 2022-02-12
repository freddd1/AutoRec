from torch import nn
class AutoRec(nn.Module):
    """
    AutoRec model.
    """
    def __init__(self, num_features, num_hidden=500):
        """
        :param num_hidden: Size of the hidden layer
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_hidden, num_features),

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded