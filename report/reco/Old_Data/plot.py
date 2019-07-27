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
        # path_dat='ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle',
        # model_list=['Vanilla', 'Sparse', 'Contractive', 'Variational'],
        path_dat='ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle',
        model_list=['Vanilla',],
        path_ml = 'ReducedFeatures/minmaxscalar',
        ml_list = ['Isolation_Forest', 'OneClass-SVM'],
        datafraction_list=[1.00 for i in range(10)]
        ):
    path_eval = os.path.join('logs', path_dat, 'eval')
    data_roc_auc = pd.read_csv(os.path.join(path_eval, 'roc_auc.txt'), sep=" ")
    model_roc_auc = {
            model: list(data_roc_auc.query('model_name == "{}"'.format(model))['roc_auc'])
            for model in model_list 
        }
    for ml in ml_list:
        model_roc_auc[ml] = list(pd.read_csv(os.path.join('logs',path_ml,'roc_auc.txt'), sep=" ").query('model_name == "{}"'.format(ml))['roc_auc'])
    model_roc_auc_mean = { key: np.sqrt(np.mean(np.square(roc_auc))) for key, roc_auc in model_roc_auc.items() }
    model_roc_auc_rms = { key: np.std(roc_auc) for key, roc_auc in model_roc_auc.items() }    
    df_list = []
    for model in model_list:
        df_model = pd.concat([
            pd.read_csv(os.path.join(path_eval, '{} model {} {}.txt'.format(model, index+1, dat_frac)), sep=" ", index_col=False)
            for index, dat_frac in enumerate(datafraction_list)])
        df_model['model'] = model
        df_list.append(df_model)
    for ml in ml_list:
        df_model = pd.concat([
            pd.read_csv(os.path.join('logs', path_ml, ml,'{} {} {}.txt'.format(ml, index+1, dat_frac)), sep=" ", index_col=False)
            for index, dat_frac in enumerate(datafraction_list)])
        df_model['model'] = ml
        df_list.append(df_model)
    df_all = pd.concat(df_list)
    ax = sns.lineplot(x="fpr", y="tpr", hue="model", data=df_all)
    ax.set(xlabel='FPR', ylabel='TPR', ylim=(0.6, 1.01), xlim=(-0.01, 1.01))
    labels = ["{}, AUC {:.2f} $\pm$ {:.3f}".format(model, 100.0*model_roc_auc_mean[model], 100.0*model_roc_auc_rms[model]) for model in model_list]
    for ml in ml_list:
        labels.append("{}, AUC {:.2f} $\pm$ {:.3f}".format(ml, 100.0*model_roc_auc_mean[ml], 100.0*model_roc_auc_rms[ml]))
    plt.legend(title='Autoencoder', loc='lower right', labels=labels)
    plt.title("Performance")
    plt.show(ax)

def evalSmooth(
        path_dat='ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle',
        model_list=['Vanilla', 'Sparse', 'Contractive', 'Variational'],
        COLOR_PALETES=['r','g','b','o'],
        datafraction_list=[1.00 for i in range(10)],
        n_bins=10
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
    plt.legend(["{}, AUC {:.2f} $\pm$ {:.3f}".format(model, 100.0*model_roc_auc_mean[model], 100.0*model_roc_auc_rms[model]) for model in model_list], loc="lower right", frameon=False)
    plt.ylim(0.5, 1.01)
    plt.show()


def evalSmoothML(
        path_dat='logs/ReducedFeatures/minmaxscalar',
        model_list=['Isolation_Forest', 'OneClass-SVM'],
        COLOR_PALETES=['r','g','b','o'],
        datafraction_list=[1.00 for i in range(10)],
        n_bins=20
        ):
    data_roc_auc = pd.read_csv(os.path.join(path_dat, 'roc_auc.txt'), sep=" ")
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
            pd.read_csv(os.path.join(path_dat, model, '{} {} {}.txt'.format(model, index+1, dat_frac)), sep=" ", index_col=False)
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
    plt.legend(["{}, AUC {:.2f} $\pm$ {:.3f}".format(model, 100.0*model_roc_auc_mean[model], 100.0*model_roc_auc_rms[model]) for model in model_list], loc="lower right", frameon=False)
    plt.ylim(0.5, 1.01)
    plt.show()

def plot_se_example(
        path_dat='logs/ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle',
        ):
    df = pd.read_csv(os.path.join(path_dat, 'SD_sample.txt'), sep=" ")
    
    good_channels = df['good_channel']
    bad_channels = df['bad_channel']
    x = range(1,len(good_channels)+1)
    
    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs[0].plot(x, good_channels)
    axs[0].set_title('Good LS ( Total SE = {:.2f} )'.format(sum(good_channels)))
    axs[0].set_xlabel("Feature Number")
    axs[0].set_ylabel("|x - $\~{x}|^2$")

    axs[1].plot(x, bad_channels)
    axs[1].set_title('Bad LS ( Total SE = {:.2f} )'.format(sum(bad_channels)))
    axs[1].set_xlabel("Feature Number")
    axs[1].set_ylabel("|x - $\~{x}|^2$")

    fig.suptitle("Example of Good and Bad LS")
    plt.show()

def plot_se_dist(
        path_dat='logs/ReducedFeatures/minmaxscalar/BS256_EP1200_Fulltrain_totalSD/total_se',
        model_name = 'Vanilla',
        n_bins = 80, # 80
        cufoff_totse = 10.5
        ):
    df_good = pd.read_csv(os.path.join(path_dat, '{}_good_totalSE.txt'.format(model_name)), sep=" ")
    df_bad = pd.read_csv(os.path.join(path_dat, '{}_bad_totalSE.txt'.format(model_name)), sep=" ")
    
    good_channels = df_good['total_se']
    bad_channels = df_bad['total_se']
    se_max = max(pd.concat([good_channels, bad_channels]))
    se_min = min(pd.concat([good_channels, bad_channels]))

    n_bad_below_cutoff = len(list(filter(lambda x: x < cufoff_totse, bad_channels)))
    percent_contamination = 100.0 * (n_bad_below_cutoff/(len(good_channels)+n_bad_below_cutoff))
    print("n_labeled {}, n_bad_below_cutoff {}, contamination {}".format(len(good_channels), n_bad_below_cutoff, percent_contamination))
    plt.figure()

    # bins = se_min + ((se_max-se_min)/n_bins * np.arange(1, n_bins+1))
    bins = 1.1**(np.arange(4,n_bins))
    plt.hist(good_channels, bins=bins, alpha=0.5, label='Labeled Good (Human)')
    plt.hist(bad_channels,  bins=bins, alpha=0.5, label='Labeled Bad (Human)')
    plt.legend(loc='upper right')
    # plt.hist([good_channels, bad_channels], n_bins, histtype='step', stacked=True, fill=False)
    plt.title('Distribution of Total Error ({})'.format(model_name))
    plt.xlabel("Total Square Error")
    plt.ylabel("#")
    plt.yscale('log')
    # plt.xlim((0, 80.0))
    plt.xscale('log')
    plt.show()

def plot_decision_val_dist(
        # path_dat='logs/ReducedFeatures/minmaxscalar/Isolation Forest',
        # model_name = 'Isolation Forest 1',
        path_dat='logs/ReducedFeatures/minmaxscalar/OneClass-SVM',
        model_name = 'OneClass-SVM 1',
        n_bins = 80, # 80
        cufoff_decision = 20.0
        ):
    df_good = pd.read_csv(os.path.join(path_dat, '{}_good_totalSE.txt'.format(model_name)), sep=" ")
    df_bad = pd.read_csv(os.path.join(path_dat, '{}_bad_totalSE.txt'.format(model_name)), sep=" ")

    ### Find % contamination
    n_good = len(df_good)
    n_bad = len(df_bad)
    n_good_contaminate = len(list(filter(lambda x: x < cufoff_decision, df_bad['total_se'])))
    percent_good_contamination = 100.0 * (n_good_contaminate/(n_good+n_good_contaminate))
    n_bad_contaminate = len(list(filter(lambda x: x > cufoff_decision, df_good['total_se'])))
    percent_bad_contamination = 100.0 * (n_bad_contaminate/(n_bad+n_bad_contaminate))
    print("Good LS contaminate {:.2f}% Bad LS contaminate {:.2f}%".format(percent_good_contamination, percent_bad_contamination))
    ###
    ### Inspect cotamination LS
    dc_contaminated_gtb, run_contaminated_gtb, ls_contaminated_gtb = df_good['total_se'][df_good['total_se'] > cufoff_decision], df_good['run'][df_good['total_se'] > cufoff_decision], df_good['lumi'][df_good['total_se'] > cufoff_decision]
    dc_contaminated_btg, run_contaminated_btg, ls_contaminated_btg = df_bad['total_se'][df_bad['total_se'] < cufoff_decision], df_bad['run'][df_bad['total_se'] < cufoff_decision], df_bad['lumi'][df_bad['total_se'] < cufoff_decision]
    print("Good falling into Bad good LS (# {})".format(len(dc_contaminated_gtb)))
    for dc, run, ls in zip(dc_contaminated_gtb, run_contaminated_gtb, ls_contaminated_gtb):
        print("DC val {} run {} ls {}".format(dc, run, ls))
    print("Bad falling into Good good LS (# {})".format(len(dc_contaminated_btg)))
    for dc, run, ls in zip(dc_contaminated_btg, run_contaminated_btg, ls_contaminated_btg):
        print("DC val {} run {} ls {}".format(dc, run, ls))
    ###
    
    good_channels = df_good['total_se']
    bad_channels = df_bad['total_se']
    se_max = max(pd.concat([good_channels, bad_channels]))
    se_min = min(pd.concat([good_channels, bad_channels]))

    plt.figure()

    bins = se_min + ((se_max-se_min)/n_bins * np.arange(-10, 80))
    # bins = se_min + ((se_max-se_min)/n_bins * np.arange(1, n_bins+1))
    # bins = 1.1**(np.arange(4,n_bins))
    plt.hist(good_channels, bins=bins, alpha=0.5, label='Labeled Good (Human)')
    plt.hist(bad_channels,  bins=bins, alpha=0.5, label='Labeled Bad (Human)')
    plt.legend(loc='upper right')
    # plt.hist([good_channels, bad_channels], n_bins, histtype='step', stacked=True, fill=False)
    plt.title('Distribution of Decision Value ({})'.format(model_name))
    plt.xlabel("Decision Value")
    plt.ylabel("#")
    plt.yscale('log')
    # plt.xlim((0, 80.0))
    # plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    plot_decision_val_dist()