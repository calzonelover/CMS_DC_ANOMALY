import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt, isnan
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


def evalTough(
        path_dat='ReducedFeatures/BS256_EP1200_Fulltrain',
        model_list=['Vanilla', 'Sparse', 'Contractive', 'Variational'],
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
    df_list = []
    for model in model_list:
        df_model = pd.concat([
            pd.read_csv(os.path.join(path_eval, '{} model {} {}.txt'.format(model, index+1, dat_frac)), sep=" ", index_col=False)
            for index, dat_frac in enumerate(datafraction_list)])
        df_model['model'] = model
        df_list.append(df_model)
    df_all = pd.concat(df_list)
    ax = sns.lineplot(x="fpr", y="tpr", hue="model", data=df_all)
    ax.set(xlabel='FPR', ylabel='TPR', ylim=(0.5, 1.01), xlim=(0.0, 1.01))
    plt.legend(title='Autoencoder', loc='lower right', labels=["{}, AUC {:.1f} $\pm$ {:.2f}".format(model, 100.0*model_roc_auc_mean[model], 100.0*model_roc_auc_rms[model]) for model in model_list])
    plt.title("Performance")
    plt.show(ax)

def evalSmooth(
        path_dat='ReducedFeatures/BS256_EP1200_Fulltrain',
        model_list=['Vanilla', 'Sparse', 'Contractive', 'Variational'],
        COLOR_PALETES=['r','g','b','o'],
        datafraction_list=[0.75 for i in range(10)],
        n_bins=50
        ):
    path_eval = os.path.join('logs', path_dat, 'eval')
    data_roc_auc = pd.read_csv(os.path.join(path_eval, 'roc_auc.txt'), sep=" ")
    model_roc_auc = {
            model: list(data_roc_auc.query('model_name == "{}"'.format(model))['roc_auc'])
            for model in model_list 
        }
    model_roc_auc_mean = { key: np.sqrt(np.mean(np.square(roc_auc))) for key, roc_auc in model_roc_auc.items() }
    model_roc_auc_rms = { key: np.std(roc_auc) for key, roc_auc in model_roc_auc.items() }
    df_list = []
    x_chunk = [i*(1.0/float(n_bins)) for i in range(n_bins+1)]
    x_middle_bins = [(float(i)+0.5)*(1.0/float(n_bins)) for i in range(n_bins)]
    y_mean_bins = {model:[] for model in model_list}
    y_rms_bins = {model:[] for model in model_list}
    for model in model_list:
        df_model = pd.concat([
            pd.read_csv(os.path.join(path_eval, '{} model {} {}.txt'.format(model, index+1, dat_frac)), sep=" ", index_col=False)
            for index, dat_frac in enumerate(datafraction_list)])
        for bin_i in range(n_bins):
            df_bin_i = df_model.query('fpr > {} & fpr < {}'.format(x_chunk[bin_i], x_chunk[bin_i+1]))
            y_mean_bin = np.mean(df_bin_i['tpr'].values)
            y_mean_bins[model].append(y_mean_bin)
            y_rms_bins[model].append(np.sqrt(np.mean(np.square(df_bin_i['tpr'].values - y_mean_bin))))
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Performance")
    for model, color in zip(model_list, COLOR_PALETES):
        plt.plot(x_middle_bins, y_mean_bins[model], label='{}'.format(model))
        plt.fill_between(
            x_middle_bins,
            np.subtract(y_mean_bins[model], y_rms_bins[model]),
            np.add(y_mean_bins[model], y_rms_bins[model]),
            alpha=0.2
            )
    plt.legend(["{}, AUC {:.1f} $\pm$ {:.2f}".format(model, 100.0*model_roc_auc_mean[model], 100.0*model_roc_auc_rms[model]) for model in model_list], loc="lower right", frameon=False)
    plt.ylim(0.5, 1.0)
    plt.show()


def plot_se(
        path_dat='logs/ReducedFeatures/BS256_EP1200_Fulltrain_perfect',
        model='Vanilla'
        ):
    df = pd.read_csv(os.path.join(path_dat, 'SD_sample.txt'), sep=" ")
    
    good_channels = df['good_channel']
    bad_channels = df['bad_channel']
    x = range(1,len(good_channels)+1)
    
    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs[0].plot(x, good_channels)
    axs[0].set_title('Good')
    axs[0].set_xlabel("Feature Number")
    axs[0].set_ylabel("|x - $\~{x}|^2$")

    axs[1].plot(x, bad_channels)
    axs[1].set_title('Anomaly')
    axs[1].set_xlabel("Feature Number")
    axs[1].set_ylabel("|x - $\~{x}|^2$")

    fig.suptitle("Example of Inlier and Anomaly")
    plt.show()

if __name__ == "__main__":
    plot_se()