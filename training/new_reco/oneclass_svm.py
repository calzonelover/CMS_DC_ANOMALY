import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from data.new_prompt_reco.setting import ( EXTENDED_FEATURES, FEATURES, FRAC_VALID, FRAC_TEST,
                                            PD_GOOD_DATA_DIRECTORY, PD_BAD_DATA_DIRECTORY )
import data.new_prompt_reco.utility as utility


COLORS = ('green', 'blue')
GROUP_LABELS = ('A', 'B')
HUMAN_LABELS = ('Good', 'Bad')

def main():    
    # setting
    model_name = "OneClass_SVM"
    selected_pd = "SingleMuon"
    print(selected_pd)
    data_preprocessing_mode = 'minmaxscalar'
    DATA_SPLIT_TRAIN = [1.0, ] # [0.2, 0.4, 0.6, 0.8, 1.0]
    is_fillna_zero = True

    features = utility.get_full_features(selected_pd)
    df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY)
    df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
    if is_fillna_zero:
        df_good = df_good.fillna(0)
        df_bad = df_bad.fillna(0)
    x = df_good[features]
    x_train_full, x_valid, x_test_good = utility.split_dataset(x, frac_test=FRAC_TEST, frac_valid=FRAC_VALID)
    y_test = np.concatenate((np.full(x_test_good.shape[0], 0), np.full(df_bad[features].shape[0], 1)))
    x_test = np.concatenate([x_test_good, df_bad[features].to_numpy()])

    file_auc = open('report/reco/eval/roc_auc.txt', 'w')
    file_auc.write("model_name data_fraction roc_auc\n")

    model_list = [svm.OneClassSVM(
    nu=0.1, kernel="rbf", gamma=0.1
    )for i in range(len(DATA_SPLIT_TRAIN))]

    for dataset_fraction, model in zip(DATA_SPLIT_TRAIN, model_list):
        print("Model: {}, Chunk of Training Dataset fraction: {}".format(model_name, dataset_fraction))
        x_train = x_train_full[:int(dataset_fraction*len(x_train_full))]
        print("Data # training: {}, # validation: {}, # testing good {}, # testing bad {}".format(
            x_train.shape[0],
            x_valid.shape[0],
            x_test_good.shape[0],
            df_bad[features].shape[0],
        ))
        # Data Preprocessing
        if data_preprocessing_mode == 'standardize':
            transformer = StandardScaler()
        elif data_preprocessing_mode == 'minmaxscalar':
            transformer = MinMaxScaler(feature_range=(0,1))
        if data_preprocessing_mode == 'normalize':
            x_train = normalize(x_train, norm='l1')
            x_valid = normalize(x_valid, norm='l1')
            x_test = normalize(x_test, norm='l1')
        else:
            transformer.fit(x_train)
            x_train_tf = transformer.transform(x_train)
            x_valid_tf = transformer.transform(x_valid)
            x_test_tf = transformer.transform(x_test)

        model.fit(x_train_tf)
        try:
            file_eval = open('report/reco/eval/{} {}.txt'.format(model_name, dataset_fraction), 'w')
        except FileNotFoundError:
            os.makedirs("./report/reco/eval/")
            file_eval = open('report/reco/eval/{} {}.txt'.format(model_name, dataset_fraction), 'w')
        file_eval.write("fpr tpr threshold\n")
        fprs, tprs, thresholds = roc_curve(y_test, -model.decision_function(x_test_tf))
        for fpt, tpr, threshold in zip(fprs, tprs, thresholds):
            file_eval.write("{} {} {}\n".format(fpt, tpr, threshold))
        file_eval.close()

        print("AUC {}".format(auc(fprs, tprs)))
        file_auc.write("{} {} {}\n".format(model_name, dataset_fraction, auc(fprs, tprs)))

    # Visualization section
    pca = PCA(n_components=2)
    pca.fit(np.concatenate([transformer.transform(df_good[features].to_numpy()), transformer.transform(df_bad[features].to_numpy())]))
    # visulize human
    x_labeled_good = pca.transform(df_good[features].to_numpy())
    x_labeled_bad = pca.transform(df_bad[features].to_numpy())
    fig, ax = plt.subplots()
    for color, x, group_label in zip(COLORS, [x_labeled_good, x_labeled_bad], GROUP_LABELS):
        ax.scatter(
            x[:, 0], x[:, 1], alpha=0.8,
            c = color,
            label = group_label
        )
    ax.legend()
    plt.title('Labeled by Human ({})'.format(selected_pd))
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.savefig('{}_label.png'.format(selected_pd))