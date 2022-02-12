import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.autorec.model import AutoRec
from src.autorec.dataset import I_AutoRecDataSet


class AutoRecTrainer():
    def __init__(self,
                 train: pd.DataFrame,
                 val: pd.DataFrame,
                 model: AutoRec,
                 batch_size: int = 112,
                 epochs: int = 10,
                 lr: float = 0.001,
                 reg: float = 0.001,
                 seed: int = 14):
        """
        Init function
        :param train: pd.DataFrame matrix where users in each row and items in each col
        :param val: pd.DataFrame matrix where users in each row and items in each col
        :param model: AutoRec model
        :param batch_size: batch size
        :param epochs: number of epochs
        :param lr: learning rate
        :param reg: lambda for regulation
        :param seed: random seed for shuffling the data set
        """

        # Set random seed
        torch.manual_seed(seed)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Train parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.reg = reg

        # Init datasets
        self.train = I_AutoRecDataSet(train)
        self.val = I_AutoRecDataSet(val)
        self.train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val, batch_size=self.batch_size, shuffle=True)

        # Init optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.reg)

        # Save the losses
        self.train_losses = []
        self.val_losses = []

    def train_model(self):

        for epoch in range(1, self.epochs + 1):
            # Train
            self.train_epoch()

            # Evaluate
            self.eval_epoch()

            # Print epoch statistics
            train_avg = np.nanmean(self.train_losses[-1])
            val_avg = np.nanmean(self.val_losses[-1])
            print(f'EPOCH {epoch}: Avg losses: train: {train_avg:.3f}, val: {val_avg:.3f}')

    def train_epoch(self):
        """
        The function will train the model for 1 epoch and save the epoch losses to `self.train_losses`
        """
        epoch_losses = []

        # set the model to train mode
        self.model.train()

        for batch_id, (rating_batch, mask_batch) in enumerate(self.train_loader):
            # Set data
            rating_batch = rating_batch.float().to(self.device)
            mask_batch = mask_batch.to(self.device)

            # Predict
            preds = self.model(rating_batch).to(self.device)

            # Loss
            train_loss = self.loss_func(preds, rating_batch, mask_batch)
            epoch_losses.append(train_loss.item())

            # backpropagation
            self.optimizer.zero_grad()
            train_loss.backward()

            # update
            self.optimizer.step()

        # save the epoch losses
        self.train_losses.append(epoch_losses)

    def eval_epoch(self):
        """
        The function will evaluate the model for 1 epoch and save the epoch losses to `self.val_losses`
        """
        epoch_val_losses = []

        # Set the model to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for batch_id, (rating_batch, mask_batch) in enumerate(self.val_loader):
                # Set data
                rating_batch = rating_batch.float().to(self.device)
                mask_batch = mask_batch.to(self.device)

                # Predict
                preds = self.model(rating_batch).to(self.device)

                # Loss
                val_loss = self.loss_func(preds, rating_batch, mask_batch)
                epoch_val_losses.append(val_loss.item())

        # save the epoch losses
        self.val_losses.append(epoch_val_losses)

    def loss_func(self, preds: torch.Tensor, rating: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        This is the loss function of the model.
        We will use RMSE
        :param preds: predictions of the model
        :param rating: the real dataset (train)
        :param mask: mask that indicates where there is rating. I.e. points that we will calculate.
        :return:
        """
        preds = preds[mask]
        rating = rating[mask]
        rmse = torch.sqrt(nn.functional.mse_loss(preds, rating))
        return rmse


