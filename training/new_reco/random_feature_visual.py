import matplotlib.pyplot as plt

import random
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
    selected_pds = ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]
    data_preprocessing_mode = 'minmaxscalar'
    BS = 2**15
    EPOCHS = 1200
    is_fillna_zero = True

    for selected_pd in selected_pds:
        features = utility.get_full_features(selected_pd)
        df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY)
        df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
        if is_fillna_zero:
            df_good_nan = pd.isnull(df_good)
            print(df_good[df_good_nan].index.tolist())
            df_good = df_good.fillna(0)
            df_bad = df_bad.fillna(0)
        x = df_good[features]
        x_train_full, x_valid, x_test_good = utility.split_dataset(x, frac_test=FRAC_TEST, frac_valid=FRAC_VALID)
        y_test = np.concatenate((np.full(x_test_good.shape[0], 0), np.full(df_bad[features].shape[0], 1)))
        x_test = np.concatenate([x_test_good, df_bad[features].to_numpy()])

        file_auc = open('report/reco/eval/roc_auc.txt', 'w')
        file_auc.write("model_name data_fraction roc_auc\n")

        x_train = x_train_full
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
            x_train = transformer.transform(x_train)
            x_valid = transformer.transform(x_valid)
            x_test = transformer.transform(x_test)

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
        plt.savefig('{}_label.png'.format(selected_pd), bbox_inches='tight')
        # random visual
        for rand_i in range(2):
            print("rand number {}".format(rand_i))
            rand_two_features = random.sample(features, 2)
            x_labeled_good = df_good[rand_two_features].to_numpy()
            x_labeled_bad = df_bad[rand_two_features].to_numpy()
            fig, ax = plt.subplots()
            for color, x, group_label in zip(COLORS, [x_labeled_good, x_labeled_bad], GROUP_LABELS):
                ax.scatter(
                    x[:, 0], x[:, 1], alpha=0.8,
                    c = color,
                    label = group_label
                )
            ax.legend()
            plt.title('Labeled by Human ({})'.format(selected_pd))
            plt.xlabel("{}".format(rand_two_features[0]))
            plt.ylabel("{}".format(rand_two_features[1]))
            plt.savefig('{}_label_rand_{}.png'.format(selected_pd, rand_i), bbox_inches='tight')