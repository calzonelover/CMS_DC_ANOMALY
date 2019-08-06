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

from data.new_prompt_reco import setting as new_prompt_reco_setting
from data.new_prompt_reco import utility as new_prompt_reco_utility

COLORS_SEPARATE = ('green', 'orange', 'red', 'purple', 'blue')
HUMAN_LABELS_SEPARATE = ('Good', 'Bad_HCAL', 'Bad_ECAL','Bad_TRACKER', 'Bad_MUON')
MARKERS = ('o', '^', '^', '^', '^')

def main(
        selected_pd = "JetHT",
        interested_statuses = {
            'hcal_hcal': 'hcal-hcal',
            'ecal_ecal': 'ecal-ecal',
            'tracker_track': 'tracker-track',
            'muon_muon': 'muon-muon'
        },        
    ):
    # setting
    data_preprocessing_mode = 'minmaxscalar'
    is_separate_plot_failure = True
    is_dropna = True
    is_fillna_zero = True

    print("\n\n Processing {} \n\n".format(selected_pd))
    features = new_prompt_reco_utility.get_full_features(selected_pd)
    df_good = new_prompt_reco_utility.read_data(selected_pd=selected_pd, pd_data_directory=new_prompt_reco_setting.PD_GOOD_DATA_DIRECTORY)
    if not is_separate_plot_failure:
        pass
    else:
        df_bad = new_prompt_reco_utility.read_data(selected_pd=selected_pd, pd_data_directory=new_prompt_reco_setting.PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY)
        df_bad_hcal = df_bad.query('hcal_hcal == 1')
        df_bad_ecal = df_bad.query('ecal_ecal == 1')
        df_bad_traker = df_bad.query('tracker_track == 1')
        df_bad_muon = df_bad.query('muon_muon == 1')
        print("Before dropna; # Good:{} , # Bad:{}, # HCAL:{}, # ECAL:{}, # TRACKER:{}, # MUON:{}".format(
            df_good.shape[0], df_bad.shape[0], df_bad_hcal.shape[0], df_bad_ecal.shape[0], df_bad_traker.shape[0], df_bad_muon.shape[0]
        ))
        if is_dropna:
            df_good = df_good.dropna()
            df_bad_hcal = df_bad_hcal.dropna()
            df_bad_ecal = df_bad_ecal.dropna()
            df_bad_traker = df_bad_traker.dropna()
            df_bad_muon = df_bad_muon.dropna()
        elif is_fillna_zero:
            df_good = df_good.fillna(0)
            df_bad_hcal = df_bad_hcal.fillna(0)
            df_bad_ecal = df_bad_ecal.fillna(0)
            df_bad_traker = df_bad_traker.fillna(0)
            df_bad_muon = df_bad_muon.fillna(0)
        x = df_good[features]
        x_train_full, x_valid, x_test_good = new_prompt_reco_utility.split_dataset(
                                                x,
                                                frac_test=new_prompt_reco_setting.FRAC_TEST,
                                                frac_valid=new_prompt_reco_setting.FRAC_VALID
                                            )
        y_test = np.concatenate((
            np.full(x_test_good.shape[0], 0),
            np.full(df_bad_hcal[features].shape[0], 1),
            np.full(df_bad_ecal[features].shape[0], 1),
            np.full(df_bad_traker[features].shape[0], 1),
            np.full(df_bad_muon[features].shape[0], 1),
        ))
        x_test = np.concatenate([
            x_test_good,
            df_bad_hcal[features].to_numpy(),
            df_bad_ecal[features].to_numpy(),
            df_bad_traker[features].to_numpy(),
            df_bad_muon[features].to_numpy(),
        ])

        file_auc = open('report/reco/eval/roc_auc.txt', 'w')
        file_auc.write("model_name data_fraction roc_auc\n")

        x_train = x_train_full
        print("Before dropna; # Good:{}, # HCAL:{}, # ECAL:{}, # TRACKER:{}, # MUON:{}".format(
            df_good.shape[0], df_bad_hcal.shape[0], df_bad_ecal.shape[0], df_bad_traker.shape[0], df_bad_muon.shape[0]
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
        # pca.fit(transformer.transform(df_good[features].to_numpy()))
        pca.fit(np.concatenate([
            transformer.transform(df_good[features].to_numpy()),
            transformer.transform(df_bad[features].to_numpy()),
        ]))
        # visualize human
        x_labeled_good = pca.transform(transformer.transform(df_good[features].to_numpy()))
        x_labeled_bad_hcal = pca.transform(transformer.transform(df_bad_hcal[features].to_numpy()))
        x_labeled_bad_ecal = pca.transform(transformer.transform(df_bad_ecal[features].to_numpy()))
        x_labeled_bad_tracker = pca.transform(transformer.transform(df_bad_traker[features].to_numpy()))
        x_labeled_bad_muon = pca.transform(transformer.transform(df_bad_muon[features].to_numpy()))
        fig, ax = plt.subplots()
        for color, x, group_label, marker in zip(
                COLORS_SEPARATE,
                [x_labeled_good, x_labeled_bad_hcal, x_labeled_bad_ecal, x_labeled_bad_tracker, x_labeled_bad_muon, ],
                HUMAN_LABELS_SEPARATE, MARKERS
            ):
            ax.scatter(
                x[:, 0], x[:, 1], alpha=0.2,
                c = color,
                marker = marker,
                label = group_label
            )
        ax.legend()
        plt.title('Labeled 2018 data ({})'.format(selected_pd))
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.savefig('{}_label.png'.format(selected_pd), bbox_inches='tight')
        plt.ylim((-3,3))
        plt.xlim((-3,3))
        plt.savefig('{}_label_short_range.png'.format(selected_pd), bbox_inches='tight')