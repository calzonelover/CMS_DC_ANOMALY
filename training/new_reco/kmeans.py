import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
import pandas as pd
import os
import json

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from data.new_prompt_reco.setting import ( EXTENDED_FEATURES, FEATURES, FRAC_VALID, FRAC_TEST,
                                            PD_GOOD_DATA_DIRECTORY, PD_BAD_DATA_DIRECTORY,
                                            PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY, PD_LABELED_SUBSYSTEM_GOOD_DATA_DIRECTORY,
                                            PD_DCS_BAD_DATA_DIRECTORY,PD_FAILURE_DATA_DIRECTORY )
import data.new_prompt_reco.utility as utility

def plot_human_label(
        selected_pds = ["ZeroBias", "JetHT", "EGamma", "SingleMuon"],
        data_preprocessing_mode = 'minmaxscalar',
        is_dropna = True,
        is_fillna_zero = True,
    ):
    # styling
    COLORS = ('green', 'blue')
    GROUP_LABELS = ('A', 'B')
    HUMAN_LABELS = ('Good', 'Bad')

    for selected_pd in selected_pds:
        print("\n\n Processing {} \n\n".format(selected_pd))
        features = utility.get_full_features(selected_pd)
        df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY)
        df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
        if is_dropna:
            df_good = df_good.dropna()
            df_bad = df_bad.dropna()
        elif is_fillna_zero:
            df_good = df_good.fillna(0)
            df_bad = df_bad.fillna(0)
        x = df_good[features]
        x_train_full, x_valid, x_test_good = utility.split_dataset(x, frac_test=FRAC_TEST, frac_valid=FRAC_VALID)
        y_test = np.concatenate((np.full(x_test_good.shape[0], 0), np.full(df_bad[features].shape[0], 1)))
        x_test = np.concatenate([x_test_good, df_bad[features].to_numpy()])

        x_train = x_train_full
        print("Data # training: {}, # validation: {}, # testing good {}, # testing bad {}".format(
            x_train.shape[0],
            x_valid.shape[0],
            x_test_good.shape[0],
            df_bad.shape[0],
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
        x_labeled_good = pca.transform(transformer.transform(df_good[features].to_numpy()))
        x_labeled_bad = pca.transform(transformer.transform(df_bad[features].to_numpy()))
        fig, ax = plt.subplots()
        for color, x, group_label in zip(COLORS, [x_labeled_good, x_labeled_bad], HUMAN_LABELS):
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



def plot_bad_good_separate_case(
        selected_pds =  ["JetHT", "ZeroBias", ],
        data_preprocessing_mode = 'minmaxscalar',
        is_dropna = True,
        is_fillna_zero = True,
    ):
    # styling
    COLORS_SEPARATE = ('green', 'red', 'purple', 'orange')
    HUMAN_LABELS_SEPARATE = ('Good', 'Bad_Human', 'Bad_FailureScenario', 'Bad_DCS')
    MARKERS = ('o', '^', '^', '^')

    for selected_pd in selected_pds:
        print("\n\n Processing {} \n\n".format(selected_pd))
        features = utility.get_full_features(selected_pd)
        df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY)
        df_bad_human = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
        df_bad_failure = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_FAILURE_DATA_DIRECTORY)
        df_bad_dcs = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_DCS_BAD_DATA_DIRECTORY)
        if is_dropna:
            df_good = df_good.dropna()
            df_bad_human = df_bad_human.dropna()
            df_bad_failure = df_bad_failure.dropna()
            df_bad_dcs = df_bad_dcs.dropna()
        elif is_fillna_zero:
            df_good = df_good.fillna(0)
            df_bad_human = df_bad_human.fillna(0)
            df_bad_failure = df_bad_failure.fillna(0)
            df_bad_dcs = df_bad_dcs.fillna(0)
        x = df_good[features]
        x_train_full, x_valid, x_test_good = utility.split_dataset(x, frac_test=FRAC_TEST, frac_valid=FRAC_VALID)
        y_test = np.concatenate((
            np.full(x_test_good.shape[0], 0),
            np.full(df_bad_human[features].shape[0], 1),
            np.full(df_bad_dcs[features].shape[0], 1)
            ))
        x_test = np.concatenate([
            x_test_good,
            df_bad_human[features].to_numpy(),
            df_bad_failure[features].to_numpy(),
            df_bad_dcs[features].to_numpy(),
            ])

        x_train = x_train_full
        print("Data # training: {}, # validation: {}, # testing good {}, # testing bad_human {}, # testing bad_failure {}, # testing bad DCS {}".format(
            x_train.shape[0],
            x_valid.shape[0],
            x_test_good.shape[0],
            df_bad_human.shape[0],
            df_bad_failure.shape[0],
            df_bad_dcs.shape[0],
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
            transformer.transform(df_bad_human[features].to_numpy()),
            transformer.transform(df_bad_dcs[features].to_numpy()),
        ]))
        # visulize human
        x_labeled_good = pca.transform(transformer.transform(df_good[features].to_numpy()))
        x_labeled_bad_human = pca.transform(transformer.transform(df_bad_human[features].to_numpy()))
        x_labeled_bad_failure = pca.transform(transformer.transform(df_bad_failure[features].to_numpy()))
        x_labeled_bad_dcs = pca.transform(transformer.transform(df_bad_dcs[features].to_numpy()))
        fig, ax = plt.subplots()
        for color, x, group_label, marker in zip(COLORS_SEPARATE,
                                        [x_labeled_good, x_labeled_bad_human, x_labeled_bad_failure, x_labeled_bad_dcs, ],
                                        HUMAN_LABELS_SEPARATE, MARKERS):
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
        plt.savefig('{}_label_separate.png'.format(selected_pd), bbox_inches='tight')
        plt.ylim((-3,3))
        plt.xlim((-3,3))
        plt.savefig('{}_label_separate_short_range.png'.format(selected_pd), bbox_inches='tight')



def plot_subsystem(
        selected_pd = "JetHT",
        interested_statuses = {
            'hcal_hcal': 'hcal-hcal',
            'ecal_ecal': 'ecal-ecal',
            'tracker_track': 'tracker-track',
            'muon_muon': 'muon-muon'
        },        
        data_preprocessing_mode = 'minmaxscalar',
        is_dropna = True,
        is_fillna_zero = True,
    ):
    # styling
    COLORS_SEPARATE = ('green', 'orange', 'red', 'purple', 'c')
    HUMAN_LABELS_SEPARATE = ('Good', 'Bad_HCAL', 'Bad_ECAL','Bad_TRACKER', 'Bad_MUON')
    MARKERS = ('o', '^', '^', '^', '^')

    print("\n\n Processing {} \n\n".format(selected_pd))
    features = utility.get_full_features(selected_pd)
    df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_LABELED_SUBSYSTEM_GOOD_DATA_DIRECTORY)
    df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY)
    df_bad_hcal = df_bad.query('hcal_hcal == 0')
    df_bad_ecal = df_bad.query('ecal_ecal == 0')
    df_bad_traker = df_bad.query('tracker_track == 0')
    df_bad_muon = df_bad.query('muon_muon == 0')
    df_bad_human = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
    df_bad_dcs = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_DCS_BAD_DATA_DIRECTORY)
    print("Before dropna; # Good:{} , # Bad:{}, # HCAL:{}, # ECAL:{}, # TRACKER:{}, # MUON:{}".format(
        df_good.shape[0], df_bad.shape[0], df_bad_hcal.shape[0], df_bad_ecal.shape[0], df_bad_traker.shape[0], df_bad_muon.shape[0]
    ))
    if is_dropna:
        df_good = df_good.dropna()
        df_bad = df_bad.dropna()
        df_bad_hcal = df_bad_hcal.dropna()
        df_bad_ecal = df_bad_ecal.dropna()
        df_bad_traker = df_bad_traker.dropna()
        df_bad_muon = df_bad_muon.dropna()

        df_bad_human = df_bad_human.dropna()
        df_bad_dcs = df_bad_dcs.dropna()
    elif is_fillna_zero:
        df_good = df_good.fillna(0)
        df_bad = df_bad.fillna(0)
        df_bad_hcal = df_bad_hcal.fillna(0)
        df_bad_ecal = df_bad_ecal.fillna(0)
        df_bad_traker = df_bad_traker.fillna(0)
        df_bad_muon = df_bad_muon.fillna(0)

        df_bad_human = df_bad_human.fillna(0)
        df_bad_dcs = df_bad_dcs.fillna(0)
    x = df_good[features]
    x_train_full, x_valid, x_test_good = utility.split_dataset(
                                            x,
                                            frac_test=FRAC_TEST,
                                            frac_valid=FRAC_VALID
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
        transformer.transform(df_bad_human[features].to_numpy()),
        transformer.transform(df_bad_dcs[features].to_numpy()),
    ]))

    ###
    print(pca.explained_variance_ratio_)
    ## For check inlier and outlier
    # filter_above_muon_malfunc = list(map(lambda x: True if x > 1.0 else False, pca.transform(transformer.transform(df_bad_muon[features].to_numpy()))[:, 1]))
    # filter_below_muon_malfunc = list(map(lambda x: True if x < 1.0 else False, pca.transform(transformer.transform(df_bad_muon[features].to_numpy()))[:, 1]))
    # print("Shape df_bad_muon before cut", df_bad_muon.shape)
    # print("Shape df_bad_muon outlier", df_bad_muon[filter_above_muon_malfunc].shape)
    # print("Shape df_bad_muon inlier", df_bad_muon[filter_below_muon_malfunc].shape)
    # print("Sample muon outlier \n", df_bad_muon[filter_above_muon_malfunc].sample(n=10)[['runId', 'lumiId']])
    # print("Sample muon inlier \n", df_bad_muon[filter_below_muon_malfunc].sample(n=10)[['runId', 'lumiId']])

    ## Component in eigen vector
    # N_FIRST_COMPONENT = 20
    # abs_st_components = list(map(lambda component, feature: {'feature': feature, 'component': component}, abs(pca.components_[0]), features))
    # sorted_abs_st_components = sorted(abs_st_components, key = lambda i: i['component'], reverse=True)
    # df_pc1 = pd.DataFrame(sorted_abs_st_components)
    # df_pc1['axis'] = 1
    # abs_nd_components = list(map(lambda component, feature: {'feature': feature, 'component': component}, abs(pca.components_[1]), features))
    # sorted_abs_nd_components = sorted(abs_nd_components, key = lambda i: i['component'], reverse=True)
    # df_pc2 = pd.DataFrame(sorted_abs_nd_components)
    # df_pc2['axis'] = 2 

    # df_pc = pd.concat([df_pc1, df_pc2], ignore_index=True)
    # df_pc.to_csv("pc_{}.csv".format(selected_pd))
    ###

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
    plt.savefig('{}_subsystem_label.png'.format(selected_pd), bbox_inches='tight')
    plt.ylim((-3,3))
    plt.xlim((-3,3))
    plt.savefig('{}_subsystem_label_short_range.png'.format(selected_pd), bbox_inches='tight')


def plot_subsystem3d(
        selected_pd = "JetHT",
        interested_statuses = {
            'hcal_hcal': 'hcal-hcal',
            'ecal_ecal': 'ecal-ecal',
            'tracker_track': 'tracker-track',
            'muon_muon': 'muon-muon'
        },        
        data_preprocessing_mode = 'minmaxscalar',
        is_dropna = True,
        is_fillna_zero = True,
    ):
    # styling
    COLORS_SEPARATE = ('green', 'orange', 'red', 'purple', 'c')
    HUMAN_LABELS_SEPARATE = ('Good', 'Bad_HCAL', 'Bad_ECAL','Bad_TRACKER', 'Bad_MUON')
    MARKERS = ('o', '^', '^', '^', '^')

    print("\n\n Processing {} \n\n".format(selected_pd))
    features = utility.get_full_features(selected_pd)
    df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_LABELED_SUBSYSTEM_GOOD_DATA_DIRECTORY)
    df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY)
    df_bad_hcal = df_bad.query('hcal_hcal == 0')
    df_bad_ecal = df_bad.query('ecal_ecal == 0')
    df_bad_traker = df_bad.query('tracker_track == 0')
    df_bad_muon = df_bad.query('muon_muon == 0')
    df_bad_human = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
    df_bad_dcs = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_DCS_BAD_DATA_DIRECTORY)
    print("Before dropna; # Good:{} , # Bad:{}, # HCAL:{}, # ECAL:{}, # TRACKER:{}, # MUON:{}".format(
        df_good.shape[0], df_bad.shape[0], df_bad_hcal.shape[0], df_bad_ecal.shape[0], df_bad_traker.shape[0], df_bad_muon.shape[0]
    ))
    if is_dropna:
        df_good = df_good.dropna()
        df_bad = df_bad.dropna()
        df_bad_hcal = df_bad_hcal.dropna()
        df_bad_ecal = df_bad_ecal.dropna()
        df_bad_traker = df_bad_traker.dropna()
        df_bad_muon = df_bad_muon.dropna()

        df_bad_human = df_bad_human.dropna()
        df_bad_dcs = df_bad_dcs.dropna()
    elif is_fillna_zero:
        df_good = df_good.fillna(0)
        df_bad = df_bad.fillna(0)
        df_bad_hcal = df_bad_hcal.fillna(0)
        df_bad_ecal = df_bad_ecal.fillna(0)
        df_bad_traker = df_bad_traker.fillna(0)
        df_bad_muon = df_bad_muon.fillna(0)

        df_bad_human = df_bad_human.fillna(0)
        df_bad_dcs = df_bad_dcs.fillna(0)
    x = df_good[features]
    x_train_full, x_valid, x_test_good = utility.split_dataset(
                                            x,
                                            frac_test=FRAC_TEST,
                                            frac_valid=FRAC_VALID
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
    pca = PCA(n_components=3)
    # pca.fit(transformer.transform(df_good[features].to_numpy()))
    pca.fit(np.concatenate([
        transformer.transform(df_good[features].to_numpy()),
        transformer.transform(df_bad_human[features].to_numpy()),
        transformer.transform(df_bad_dcs[features].to_numpy()),
    ]))

    # visualize human
    x_labeled_good = pca.transform(transformer.transform(df_good[features].to_numpy()))
    x_labeled_bad_hcal = pca.transform(transformer.transform(df_bad_hcal[features].to_numpy()))
    x_labeled_bad_ecal = pca.transform(transformer.transform(df_bad_ecal[features].to_numpy()))
    x_labeled_bad_tracker = pca.transform(transformer.transform(df_bad_traker[features].to_numpy()))
    x_labeled_bad_muon = pca.transform(transformer.transform(df_bad_muon[features].to_numpy()))
    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for color, x, group_label, marker in zip(
            COLORS_SEPARATE,
            [x_labeled_good, x_labeled_bad_hcal, x_labeled_bad_ecal, x_labeled_bad_tracker, x_labeled_bad_muon, ],
            HUMAN_LABELS_SEPARATE, MARKERS
        ):
        ax.scatter(
            x[:, 0], x[:, 1], x[:, 2], alpha=0.2,
            c = color,
            marker = marker,
            label = group_label
        )
    ax.legend()
    plt.title('Labeled 2018 data ({})'.format(selected_pd))
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.savefig('{}_subsystem_label.png'.format(selected_pd), bbox_inches='tight')
    # plt.ylim((-3,3))
    # plt.xlim((-3,3))
    # plt.savefig('{}_subsystem_label_short_range.png'.format(selected_pd), bbox_inches='tight')
    for azimuth in [0, 45, 90, 135, 180]:
        for phi in [0, 45, 90, 135, 180]:
            ax.view_init(azimuth, phi)
            plt.savefig('{}_subsystem_label_short_range({}{}).png'.format(selected_pd, azimuth, phi))