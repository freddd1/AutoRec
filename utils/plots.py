import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_train_history(train_loss:List[List[float]], val_loss:List[List[float]]) -> None:
    """
    The function will print train and test avg loss per epoch
    :param train_loss: List[List[float]] of train history, may include nans
    :param val_loss: List[List[float]] of val history, may include nans
    :return: None. Will print the train graph
    """
    train_loss = np.nanmean(train_loss, axis=1)
    val_loss = np.nanmean(val_loss, axis=1)

    plt.figure(figsize=(10,6))

    plt.plot(train_loss)
    plt.plot(val_loss)

    plt.legend(['train', 'val'], loc='upper right')
    plt.title(f'RMSE Loss History', fontsize=18)
    plt.xlabel('Epochs')
    plt.ylabel(f'RMSE')
    plt.show()