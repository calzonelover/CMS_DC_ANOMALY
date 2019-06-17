import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils import shuffle
# customize
from data.prompt_reco.setting import FEATURES, SELECT_PD
import data.prompt_reco.utility as utility
from model.reco.autoencoder import VariationalAutoencoder as Autoencoder

def main():
    # setting
    BS = 64
    EPOCHS = 3000
    MODEL_NAME = "Variational"
    # data
    files = utility.get_file_list(chosed_pd=SELECT_PD) # choosing only ZeroBias

    feature_names = utility.get_feature_name()

    data = pd.DataFrame(utility.get_data(files), columns=feature_names)
    data["run"] = data["run"].astype(int)
    data["lumi"] = data["lumi"].astype(int)
    data.drop(["_foo", "_bar", "_baz"], axis=1, inplace=True)
    data = data.sort_values(["run", "lumi"], ascending=[True,True])
    data = data.reset_index(drop=True)

    data["label"] = data.apply(utility.add_flags, axis=1)
    data = data.reindex(np.random.permutation(data.index))
    # model
    model_list = [
            Autoencoder(
                summary_dir = "model/reco/summary",
                model_name = "{} model {}".format(MODEL_NAME, i),
                batch_size = BS
            )
        for i in range(1,6)]
    # training
    for dataset_fraction, autoencoder in zip(np.array([0.2,0.4,0.6,0.8,1.0]), model_list):
        print("Dataset fraction: %s" % dataset_fraction)
        print("Preparing dataset...")
        split = int(dataset_fraction*len(data))    
        dataset = data.iloc[:split].copy()

        print("Train test split...")
        split = int(0.8*len(dataset))
        df_train = dataset.iloc[:split].copy()
        X_train = df_train.iloc[:, 0:2806]
        y_train = df_train["label"]
        X_train = X_train[y_train == 0]

        print("Training autoencoder")
        model_name = "SAE%s" % int(10*dataset_fraction)
        # log
        file_log = open('report/reco/logs/{}.txt'.format(autoencoder.model_name), 'w')
        file_log.write("EP loss_train loss_valid\n")
        # standardize data
        # transformer = StandardScaler()
        # transformer.fit(X_train.values)
        # X_train = transformer.transform(X_train.values)
        X_train = normalize(X_train, norm='l1')

        X_train = X_train[:int(0.75*len(X_train))]
        X_valid = X_train[int(0.75*len(X_train)):]
        # LOOP EPOCH
        autoencoder.init_variables()
        for EP in range(EPOCHS):
            X_train = shuffle(X_train)
            for iteration_i in range(int(len(X_train)/BS) - 1):
                x_batch = X_train[BS*iteration_i: BS*(iteration_i+1)]
                autoencoder.train(x_batch)
            autoencoder.log_summary(X_train, EP)
            file_log.write("{} {} {}\n".format(
                EP,
                autoencoder.get_loss(X_train)[0],
                autoencoder.get_loss(X_valid)[0]
                ))
        file_log.close()
        autoencoder.save()



def evaluation():
    pass