import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
# customize
from data.prompt_reco.setting import REDUCED_FEATURES, FEATURES, SELECT_PD
import data.prompt_reco.utility as utility

# from model.reco.autoencoder import SparseAutoencoder as Autoencoder
from model.reco.autoencoder import ( VanillaAutoencoder, SparseAutoencoder,
                                     ContractiveAutoencoder, VariationalAutoencoder )

def main():
    # setting
    is_reduced_data = True
    data_preprocessing_mode = 'minmaxscalar' # ['standardize, 'normalize', 'minmaxscalar']
    BS = 256
    EPOCHS = 1500
    SPLIT_DATA_IN_80 = [1.0 for i in range(10)] # 60% of data
    N_FEATURES = len(REDUCED_FEATURES*7) if is_reduced_data else 2807

    files = utility.get_file_list(chosed_pd=SELECT_PD) # choosing only ZeroBias
    feature_names = utility.get_feature_name(features=FEATURES)
    reduced_feature_names = utility.get_feature_name(features=REDUCED_FEATURES)
    data = pd.DataFrame(utility.get_data(files), columns=feature_names)
    data["run"] = data["run"].astype(int)
    data["lumi"] = data["lumi"].astype(int)
    data.drop(["_foo", "_bar", "_baz"], axis=1, inplace=True)
    if is_reduced_data:
        not_reduced_column = feature_names
        for intersected_elem in reduced_feature_names: not_reduced_column.remove(intersected_elem)
        data.drop(not_reduced_column, axis=1, inplace=True)
    data = data.sort_values(["run", "lumi"], ascending=[True,True])
    data = data.reset_index(drop=True)
    data["label"] = data.apply(utility.add_flags, axis=1)
    data = data.reindex(np.random.permutation(data.index))

    file_auc = open('report/reco/eval/roc_auc.txt', 'w')
    file_auc.write("model_name data_fraction roc_auc\n")
    for model_name, Autoencoder in zip(
            [ "Vanilla", "Sparse", "Contractive", "Variational"],
            [ VanillaAutoencoder, SparseAutoencoder, ContractiveAutoencoder, VariationalAutoencoder]
            ):
        MODEL_NAME = model_name
        # model
        model_list = [
                Autoencoder(
                    input_dim = [N_FEATURES],
                    summary_dir = "model/reco/summary",
                    model_name = "{} model {}".format(MODEL_NAME, i),
                    batch_size = BS
                )
            for i in range(1,len(SPLIT_DATA_IN_80) + 1)]
        # training
        for dataset_fraction, autoencoder in zip(np.array(SPLIT_DATA_IN_80), model_list):
            print("Model: {}, Chunk of Training Dataset fraction: {}".format(autoencoder.model_name, dataset_fraction))
            print("Train(Train,Valid) test split...")
            split = int(0.8*len(data))
            # train set
            df_train = data.iloc[:split].copy()
            split_frac = int(dataset_fraction*len(df_train))
            df_train_frac = data.iloc[:split_frac].copy()
            X_train = df_train_frac.iloc[:, 0:N_FEATURES]
            y_train = df_train_frac["label"]
            # test set
            df_test = data.iloc[split:].copy()
            X_test = df_test.iloc[:, 0:N_FEATURES]
            y_test = df_test["label"]
            X_test = pd.concat([X_train[y_train == 1], X_test])
            y_test = pd.concat([y_train[y_train == 1], y_test])
            # train only good condition
            X_train = X_train[y_train == 0]
            print("Number of inliers in training&valid set: {}".format(len(X_train)))
            print("Number of inliers in test set: {}".format(sum((y_test == 0).values)))
            print("Number of anomalies in the test set: {}".format(sum((y_test == 1).values)))

            print("Training {} autoencoder".format(MODEL_NAME))
            # log
            file_log = open('report/reco/logs/{}.txt'.format(autoencoder.model_name), 'w')
            file_log.write("EP loss_train loss_valid\n")

            # Data Preprocessing
            if data_preprocessing_mode == 'standardize':
                transformer = StandardScaler()
            elif data_preprocessing_mode == 'minmaxscalar':
                transformer = MinMaxScaler(feature_range=(0,1))
            transformer.fit(X_train)
            if data_preprocessing_mode == 'normalize':
                X_train = normalize(X_train, norm='l1')
                X_test = normalize(X_test, norm='l1')
            else:
                X_train = transformer.transform(X_train.values)
                X_test = transformer.transform(X_test.values)
            X_train = X_train[:int(0.75*len(X_train))]
            X_valid = X_train[int(0.75*len(X_train)):]
            # LOOP EPOCH
            autoencoder.init_variables()
            for EP in range(EPOCHS):
                X_train = shuffle(X_train)
                for iteration_i in range(int(len(X_train)/BS)):
                    x_batch = X_train[BS*iteration_i: BS*(iteration_i+1)]
                    autoencoder.train(x_batch)
                autoencoder.log_summary(X_train, EP)
                file_log.write("{} {} {}\n".format(
                    EP,
                    autoencoder.get_loss(X_train)["loss_total"],
                    autoencoder.get_loss(X_valid)["loss_total"]
                    ))
            file_log.close()
            # Evaluation
            try:
                file_eval = open('report/reco/eval/{} {}.txt'.format(autoencoder.model_name, dataset_fraction), 'w')
            except FileNotFoundError:
                os.makedirs("./report/reco/eval/")
                file_eval = open('report/reco/eval/{} {}.txt'.format(autoencoder.model_name, dataset_fraction), 'w')
            file_eval.write("fpr tpr threshold\n")
            fprs, tprs, thresholds = roc_curve(y_test, autoencoder.get_sd(X_test, scalar=True))
            for fpt, tpr, threshold in zip(fprs, tprs, thresholds):
                file_eval.write("{} {} {}\n".format(fpt, tpr, threshold))
            file_eval.close()
            
            print("AUC {}".format(auc(fprs, tprs)))
            file_auc.write("{} {} {}\n".format(MODEL_NAME, dataset_fraction, auc(fprs, tprs)))

            autoencoder.save()

    file_auc.close()

def test_ms(Autoencoder=VanillaAutoencoder, test_model="Vanilla", number_model=1):
        # setting
        data_preprocessing_mode = 'minmaxscalar'
        is_reduced_data = True
        BS = 256
        dataset_fraction = 0.75

        N_FEATURES = len(REDUCED_FEATURES*7) if is_reduced_data else 2807
        # data
        files = utility.get_file_list(chosed_pd=SELECT_PD) # choosing only ZeroBias

        feature_names = utility.get_feature_name(features=FEATURES)
        ###
        reduced_feature_names = utility.get_feature_name(features=REDUCED_FEATURES)
        ###

        data = pd.DataFrame(utility.get_data(files), columns=feature_names)
        data["run"] = data["run"].astype(int)
        data["lumi"] = data["lumi"].astype(int)
        data.drop(["_foo", "_bar", "_baz"], axis=1, inplace=True)
        ###
        if is_reduced_data:
            not_reduced_column = feature_names
            for intersected_elem in reduced_feature_names: not_reduced_column.remove(intersected_elem)
            data.drop(not_reduced_column, axis=1, inplace=True)
        ###

        data = data.sort_values(["run", "lumi"], ascending=[True,True])
        data = data.reset_index(drop=True)

        data["label"] = data.apply(utility.add_flags, axis=1)

        data = data.reindex(np.random.permutation(data.index))
        #
        autoencoder = Autoencoder(
            input_dim = [N_FEATURES],
            summary_dir = "model/reco/summary",
            model_name = "{} model {}".format(test_model, number_model),
            batch_size = BS
        )
        autoencoder.restore()
        ##
        split = int(dataset_fraction*len(data))
        dataset = data.iloc[:split].copy()

        print("Train test split...")
        split = int(0.8*len(data))
        # train set
        df_train = data.iloc[:split].copy()
        X_train = df_train.iloc[:, 0:N_FEATURES]
        y_train = df_train["label"]
        # test set
        df_test = data.iloc[split:].copy()
        X_test = df_test.iloc[:, 0:N_FEATURES]
        y_test = df_test["label"]
        X_test = pd.concat([X_train[y_train == 1], X_test])
        y_test = pd.concat([y_train[y_train == 1], y_test])
        # train only good condition
        X_train = X_train[y_train == 0]
        print("Number of inliers in training&valid set: {}".format(len(X_train)))
        print("Number of inliers in test set: {}".format(sum((y_test == 0).values)))
        print("Number of anomalies in the test set: {}".format(sum((y_test == 1).values)))
        # log
        file_log = open('report/reco/logs/{}.txt'.format(autoencoder.model_name), 'w')
        file_log.write("EP loss_train loss_valid\n")

        # Data Preprocessing
        if data_preprocessing_mode == 'standardize':
            transformer = StandardScaler()
        elif data_preprocessing_mode == 'minmaxscalar':
            transformer = MinMaxScaler(feature_range=(0,1))
        transformer.fit(X_train)
        if data_preprocessing_mode == 'normalize':
            X_train = normalize(X_train, norm='l1')
            X_test = normalize(X_test, norm='l1')
        else:
            X_train = transformer.transform(X_train.values)
            X_test = transformer.transform(X_test.values)

        x_good = X_test[y_test == 0]
        x_good_sample = x_good[12]
        x_bad = X_test[y_test == 1]
        x_bad_sample = x_bad[56]
        
        print(x_good_sample, x_bad_sample)
        print('sample', len(x_good_sample), len(x_bad_sample))

        with open('SD_sample.txt', 'w') as f:
            f.write('good_channel bad_channel\n')
            good_square_differents = autoencoder.get_sd(np.reshape(x_good_sample, [1, len(x_bad_sample)]))[0,:]
            bad_square_differents = autoencoder.get_sd(np.reshape(x_bad_sample, [1, len(x_bad_sample)]))[0,:]
            print(good_square_differents, bad_square_differents)
            print('predict', len(good_square_differents), len(bad_square_differents))
            for good_square_different, bad_square_different in zip(good_square_differents, bad_square_differents):
                f.write('{} {}\n'.format(good_square_different, bad_square_different))