import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from data.new_prompt_reco.setting import ( EXTENDED_FEATURES, FEATURES, FRAC_VALID, FRAC_TEST,
                                            PD_GOOD_DATA_DIRECTORY, PD_BAD_DATA_DIRECTORY )
import data.new_prompt_reco.utility as utility


def main():
    # setting
    model_name = "One-Class_SVM"
    selected_pd = "SingleMuon"
    data_preprocessing_mode = 'minmaxscalar'
    BS = 2**15
    EPOCHS = 1200
    DATA_SPLIT_TRAIN = [0.2, 0.4, 0.6, 0.8, 1.0]
    is_fillna_zero = True

    features = utility.get_full_features(selected_pd)
    df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY)
    df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
    if is_fillna_zero:
        df_good = df_good.fillna(0)
        df_bad = df_bad.fillna(0)
    x = df_good[features]
    x_train_full, x_valid, x_test = utility.split_dataset(x, frac_test=FRAC_TEST, frac_valid=FRAC_VALID)
    y_test = np.concatenate((np.full(x_test.shape[0], 0), np.full(df_bad[features].shape[0], 1)))
    x_test = np.concatenate([x_test, df_bad[features].to_numpy()])

    file_auc = open('report/reco/eval/roc_auc.txt', 'w')
    file_auc.write("model_name data_fraction roc_auc\n")

    model_list = [svm.OneClassSVM(
    nu=0.1, kernel="rbf", gamma=0.1
    )for i in range(10)]

    for dataset_fraction, model in zip(DATA_SPLIT_TRAIN, model_list):
        print("Model: {}, Chunk of Training Dataset fraction: {}".format(model_name, dataset_fraction))

        x_train = x_train_full[:int(dataset_fraction*len(x_train_full))]
        # Data Preprocessing
        if data_preprocessing_mode == 'standardize':
            transformer = StandardScaler()
        elif data_preprocessing_mode == 'minmaxscalar':
            transformer = MinMaxScaler(feature_range=(0,1))
        transformer.fit(x_train)
        if data_preprocessing_mode == 'normalize':
            x_train = normalize(x_train, norm='l1')
            x_valid = normalize(x_valid, norm='l1')
            x_test = normalize(x_test, norm='l1')
        else:
            x_train = transformer.transform(x_train)
            x_valid = transformer.transform(x_valid)
            x_test = transformer.transform(x_test)
        model.fit(x_train)
        try:
            file_eval = open('report/reco/eval/{} {}.txt'.format(model_name, dataset_fraction), 'w')
        except FileNotFoundError:
            os.makedirs("./report/reco/eval/")
            file_eval = open('report/reco/eval/{} {}.txt'.format(model_name, dataset_fraction), 'w')
        file_eval.write("fpr tpr threshold\n")
        fprs, tprs, thresholds = roc_curve(y_test, -model.decision_function(x_test))
        for fpt, tpr, threshold in zip(fprs, tprs, thresholds):
            file_eval.write("{} {} {}\n".format(fpt, tpr, threshold))
        file_eval.close()

        print("AUC {}".format(auc(fprs, tprs)))
        file_auc.write("{} {} {}\n".format(model_name, dataset_fraction, auc(fprs, tprs)))