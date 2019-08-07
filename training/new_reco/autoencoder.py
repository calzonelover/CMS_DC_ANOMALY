import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from data.new_prompt_reco.setting import ( FEATURE_SET_NUMBER, EXTENDED_FEATURES, FEATURES, FRAC_VALID, FRAC_TEST,
                                            PD_GOOD_DATA_DIRECTORY, PD_BAD_DATA_DIRECTORY )
import data.new_prompt_reco.utility as utility

from model.reco.new_autoencoder import ( VanillaAutoencoder, SparseAutoencoder,
                                     ContractiveAutoencoder, VariationalAutoencoder,
                                     SparseContractiveAutoencoder, SparseVariationalAutoencoder,
                                     ContractiveVariationalAutoencoder, StandardAutoencoder)


def main(selected_pd="JetHT"):
    # setting
    data_preprocessing_mode = 'minmaxscalar'
    BS = 2**15
    EPOCHS = 1800
    DATA_SPLIT_TRAIN = [1.0 for i in range(10)]
    is_fillna_zero = True

    features = utility.get_full_features(selected_pd)
    df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY)
    df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
    if is_fillna_zero:
        df_good = df_good.fillna(0.0)
        df_bad = df_bad.fillna(0.0)
    x = df_good[features]
    x_train_full, x_valid, x_test_good = utility.split_dataset(x, frac_test=FRAC_TEST, frac_valid=FRAC_VALID)
    y_test = np.concatenate([np.full(x_test_good.shape[0], 0.0), np.full(df_bad[features].shape[0], 1.0)])
    x_test = np.concatenate([x_test_good, df_bad[features].to_numpy()])

    file_auc = open('report/reco/eval/roc_auc_{}.txt'.format(selected_pd), 'w')
    file_auc.write("model_name data_fraction roc_auc\n")
    for model_name, Autoencoder in zip(
            [ "Vanilla", "Sparse", "Contractive", "Variational"], # ["SparseContractive", "SparseVariational", "ContractiveVariational", "Standard"],
            [ VanillaAutoencoder, SparseAutoencoder, ContractiveAutoencoder, VariationalAutoencoder], #[SparseContractiveAutoencoder, SparseVariationalAutoencoder, ContractiveVariationalAutoencoder, StandardAutoencoder]
            ):
        model_list = [
            Autoencoder(
                input_dim = [len(features)],
                summary_dir = "model/reco/summary",
                model_name = "{}_model_{}_f{}_{}".format(model_name, selected_pd, FEATURE_SET_NUMBER, i),
                batch_size = BS
            )
            for i in range(1,len(DATA_SPLIT_TRAIN) + 1)
        ]
        for dataset_fraction, autoencoder in zip(DATA_SPLIT_TRAIN, model_list):
            print("Model: {}, Chunk of Training Dataset fraction: {}".format(autoencoder.model_name, dataset_fraction))
            file_log = open('report/reco/logs/{}.txt'.format(autoencoder.model_name), 'w')
            file_log.write("EP loss_train loss_valid\n")

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
            autoencoder.init_variables()
            for EP in range(EPOCHS):
                x_train_shuf = shuffle(x_train_tf)
                for iteration_i in range(int(len(x_train_shuf)/BS)):
                    x_batch = x_train_shuf[BS*iteration_i: BS*(iteration_i+1)]
                    autoencoder.train(x_batch)
                autoencoder.log_summary(x_train_tf, EP)
                file_log.write("{} {} {}\n".format(
                        EP+1,
                        autoencoder.get_loss(x_train_tf)["loss_total"],
                        autoencoder.get_loss(x_valid_tf)["loss_total"]
                        ))
            file_log.close()

            try:
                file_eval = open('report/reco/eval/{} {}.txt'.format(autoencoder.model_name, dataset_fraction), 'w')
            except FileNotFoundError:
                os.makedirs("./report/reco/eval/")
                file_eval = open('report/reco/eval/{} {}.txt'.format(autoencoder.model_name, dataset_fraction), 'w')
            file_eval.write("fpr tpr threshold\n")
            ### Tracking Error
            print("Error tracking for model: {}, # NaN in SD: {}, # inf in SD: {} ".format(
                model_name,
                len(list(filter(lambda x: x == True, np.isnan(autoencoder.get_sd(x_test_tf, scalar=True))))),
                len(list(filter(lambda x: x == True, np.isinf(autoencoder.get_sd(x_test_tf, scalar=True)))))
                ))
            ###
            fprs, tprs, thresholds = roc_curve(y_test, autoencoder.get_sd(x_test_tf, scalar=True))
            for fpt, tpr, threshold in zip(fprs, tprs, thresholds):
                file_eval.write("{} {} {}\n".format(fpt, tpr, threshold))
            file_eval.close()

            print("AUC {}".format(auc(fprs, tprs)))
            file_auc.write("{} {} {}\n".format(model_name, dataset_fraction, auc(fprs, tprs)))

            autoencoder.save()

def compute_ms_dist(
        selected_pd = "JetHT",
        Autoencoder=VanillaAutoencoder,
        model_name="Vanilla",
        number_model=1
    ):
    # setting
    data_preprocessing_mode = 'minmaxscalar'
    BS = 2**15
    is_fillna_zero = True

    features = utility.get_full_features(selected_pd)
    df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY)
    df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
    if is_fillna_zero:
        df_good = df_good.fillna(0.0)
        df_bad = df_bad.fillna(0.0)
    x = df_good[features]
    x_train_full, x_valid, x_test_good = utility.split_dataset(x, frac_test=FRAC_TEST, frac_valid=FRAC_VALID)
    y_test = np.concatenate([np.full(x_test_good.shape[0], 0.0), np.full(df_bad[features].shape[0], 1.0)])
    x_test = np.concatenate([x_test_good, df_bad[features].to_numpy()])

    x_train = x_train_full

    # Data Preprocessing
    if data_preprocessing_mode == 'standardize':
        transformer = StandardScaler()
    elif data_preprocessing_mode == 'minmaxscalar':
        transformer = MinMaxScaler(feature_range=(0,1))
    if data_preprocessing_mode == 'normalize':
        x_test_good_tf = normalize(x_test_good, norm='l1') 
        x_test_bad_tf = normalize(df_bad[features].to_numpy(), norm='l1')
    else:
        transformer.fit(x_train)
        x_test_good_tf = transformer.transform(x_test_good)
        x_test_bad_tf = transformer.transform(df_bad[features].to_numpy())
    run_good, lumi_good = df_good['runId'].iloc[-int(FRAC_TEST*len(x)):].to_numpy(), df_good['lumiId'].iloc[-int(FRAC_TEST*len(x)):].to_numpy()
    run_bad, lumi_bad = df_bad['runId'].to_numpy(), df_bad['lumiId'].to_numpy()

    autoencoder = Autoencoder(
        input_dim = [len(features)],
        summary_dir = "model/reco/summary",
        model_name = "{} model {} {}".format(model_name, selected_pd, number_model),
        batch_size = BS   
    )
    autoencoder.restore()

    with open('good_totalSE_{}_{}_{}.txt'.format(model_name, selected_pd, number_model), 'w') as f:
        f.write('total_se run lumi\n')
        for good_totalsd, run, lumi in zip(autoencoder.get_sd(x_test_good_tf, scalar=True), run_good, lumi_good):
            f.write('{} {} {}\n'.format(good_totalsd, run, lumi))
    with open('bad_totalSE_{}_{}_{}.txt'.format(model_name, selected_pd, number_model), 'w') as f:
        f.write('total_se run lumi\n')
        for bad_totalsd, run, lumi in zip(autoencoder.get_sd(x_test_bad_tf, scalar=True), run_bad, lumi_bad):
            f.write('{} {} {}\n'.format(bad_totalsd, run, lumi))