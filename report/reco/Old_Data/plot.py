import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import os

# Example: RMS 
# rms = sqrt(mean_squared_error(y_actual, y_predicted))

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

def plot_roc(fprs, tprs, title):
    fig, ax = plt.subplots()
    plt.plot(
             fprs,
             tprs,
            )
    plt.legend([""], frameon=False)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.savefig('{}.png'.format(title))

def main_loss():
    # setting
    model_name = "Vanilla model"
    datdir = "BS64_EP3000"    
    for i in range(1,6):
        data = pd.read_csv('logs/{}/{} {}.txt'.format(datdir, model_name,i), sep=" ")
        # print(data['EP'].shape)
        x = data['EP']
        loss_train = data['loss_train']
        loss_valid = data['loss_valid']
        plot_loss(x, loss_train, loss_valid, title="{} {}".format(model_name, i))


def eval(
        path_dat='ReducedFeatures/BS256_EP1200_Fulltrain',
        model_list=['Vanilla', 'Sparse', 'Contractive', 'Variational'],
        n_each_model=10,
        datafraction_list=[0.75 for i in range(10)]
        ):
    path_eval = os.path.join('logs', path_dat, 'eval')
    data_roc_auc = pd.read_csv(os.path.join(path_eval, 'roc_auc.txt'), sep=" ")
    model_roc_auc = {
            model: list(data_roc_auc.query('model_name == "{}"'.format(model))['roc_auc'])
            for model in model_list 
        }
    model_roc_auc_mean = { key: np.sqrt(np.mean(np.square(roc_auc))) for key, roc_auc in model_roc_auc.items() }
    model_roc_auc_rms = { key: np.std(roc_auc) for key, roc_auc in model_roc_auc.items() }
    print(model_roc_auc, model_roc_auc_mean, model_roc_auc_rms)

if __name__ == "__main__":
    eval()