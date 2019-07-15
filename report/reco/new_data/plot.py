import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import auc
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

def plot_roc(
        channel = "JetHT",
        model_toughs = ["Vanilla"],
        model_smooths = ['Sparse', 'Contractive', 'Variational'],
        path_dat = "logs/minmaxscalar/2e15BS12000EP",
    ):
    eval_files = os.listdir(path_dat)
    for model_tough in model_toughs:
        model_tough_eval_files = list(filter(lambda x: channel in x and model_tough in x, eval_files))
        print(model_tough_eval_files)
        print(len(model_tough_eval_files))
    for model_smooth in model_smooths:
        model_smooth_eval_files = list(filter(lambda x: channel in x and model_smooth in x, eval_files))
        print(model_smooth_eval_files)
        print(len(model_smooth_eval_files))

def evalSmooth(
        channel = "JetHT",
        path_dat='logs/minmaxscalar/2e15BS12000EP',
        model_list=['Vanilla', 'Sparse', 'Contractive', 'Variational'],
        COLOR_PALETES=['r','g','b','o'],
        datafraction_list=[1.00 for i in range(10)],
        n_bins=40
        ):
    data_roc_auc = pd.read_csv(os.path.join(path_dat, 'roc_auc_{}.txt'.format(channel)), sep=" ")
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
            pd.read_csv(os.path.join(path_dat, '{} model {} {} {}.txt'.format(model, channel, index+1, dat_frac)), sep=" ", index_col=False)
            for index, dat_frac in enumerate(datafraction_list)])
        for bin_i in range(n_bins):
            df_bin_i = df_model.query('fpr > {} & fpr < {}'.format(x_chunk[bin_i], x_chunk[bin_i+1]))
            y_mean_bin = np.mean(df_bin_i['tpr'].values)
            y_mean_bins[model].append(y_mean_bin)
            y_rms_bins[model].append(np.sqrt(np.mean(np.square(df_bin_i['tpr'].values - y_mean_bin))))
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Performance {} datasets".format(channel))
    for model, color in zip(model_list, COLOR_PALETES):
        plt.plot(x_middle_bins, y_mean_bins[model], label='{}'.format(model))
        plt.fill_between(
            x_middle_bins,
            np.subtract(y_mean_bins[model], y_rms_bins[model]),
            np.add(y_mean_bins[model], y_rms_bins[model]),
            alpha=0.2
            )
    if np.min(100.0*model_roc_auc_mean[model]) > 50.0:
        plt.legend(["{}, AUC {:.2f} $\pm$ {:.3f}".format(model, 100.0*model_roc_auc_mean[model], 100.0*model_roc_auc_rms[model]) for model in model_list], loc="lower right", frameon=False)
    else:
        plt.legend(["{}, AUC {:.2f} $\pm$ {:.3f}".format(model, 100.0*model_roc_auc_mean[model], 100.0*model_roc_auc_rms[model]) for model in model_list], loc="upper left", frameon=False)
    plt.ylim(0.0, 1.01)
    # plt.show()
    plt.savefig(os.path.join(path_dat, 'performance_{}.png').format(channel))

def plot_decision_val_dist(
        path_dat='logs/minmaxscalar/2e15BS12000EP',
        channel = 'JetHT',
        model_name = 'Vanilla',
        model_number = 1,
        n_bins = 80, # 80
        base_log = 1.1
        ):
    df_good = pd.read_csv(os.path.join(path_dat, 'good_totalSE_{}_{}_{}.txt'.format(model_name, channel, model_number)), sep=" ")
    df_bad = pd.read_csv(os.path.join(path_dat, 'bad_totalSE_{}_{}_{}.txt'.format(model_name, channel, model_number)), sep=" ")
    
    good_channels = df_good['total_se']
    bad_channels = df_bad['total_se']
    se_min = min([min(good_channels), min(bad_channels)])
    se_max = max([max(good_channels), max(bad_channels)])

    fig, ax = plt.subplots()

    # bins = se_min + ((se_max-se_min)/n_bins * np.arange(0, n_bins+1))
    bins = base_log**(np.arange(2, n_bins))
    plt.hist(good_channels, bins=bins, alpha=0.5, label='Labeled Good (Human)')
    plt.hist(bad_channels,  bins=bins, alpha=0.5, label='Labeled Bad (Human)')
    plt.legend(loc='upper right')
    # plt.hist([good_channels, bad_channels], n_bins, histtype='step', stacked=True, fill=False)
    plt.title('Distribution of Decision Value ({}, {} datasets)'.format(model_name, channel))
    plt.xlabel("Total Square Error")
    plt.ylabel("#")
    plt.yscale('log')
    # plt.xlim((0, 80.0))
    plt.xscale('log')
    plt.savefig('se_dist_{}{}_{}.png'.format(model_name, model_number, channel))
    # plt.show()

if __name__ == "__main__":
    for channel, n_bins, base_log in zip(['ZeroBias', 'JetHT', 'EGamma', 'SingleMuon'], [60, 60, 80, 80], [1.05, 1.05, 1.1, 1.1]):
        plot_decision_val_dist(channel=channel, n_bins=n_bins, base_log=base_log)