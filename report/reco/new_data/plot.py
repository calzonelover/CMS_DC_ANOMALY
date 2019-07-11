import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt, isnan
import os

def plot_loss(x, loss_train, loss_valid, title):
    """Plots the training and validation loss"""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.plot(x, loss_train, '-b', label='Training')
    plt.plot(x, loss_valid, '-r', linestyle=(0, (1, 2)), label='Validation')
    plt.legend(["Training", "Validation"], loc="upper right", frameon=False)
    plt.yscale("log")
    # plt.show()
    plt.savefig('{}.png'.format(title))

def main_loss(
        channel = "JetHT",
        model_name = "Variational model",
        datdir = "logs/minmaxscalar/2e16BS12000EP",
        ):
    for i in range(1,6):
        data = pd.read_csv('{}/{}/{} {}.txt'.format(datdir, channel, model_name,i), sep=" ")
        # print(data['EP'].shape)
        x = data['EP']
        loss_train = data['loss_train']
        loss_valid = data['loss_valid']
        plot_loss(x, loss_train, loss_valid, title="{} {} ({})".format(model_name, i, channel))

if __name__ == "__main__":
    main_loss()