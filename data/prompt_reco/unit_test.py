import numpy as np
import pandas as pd
import os

# customize
from data.prompt_reco.setting import REDUCED_FEATURES, FEATURES, SELECT_PD
import data.prompt_reco.utility as utility

def main():
    # Setting
    is_reduced_data = True
    N_FEATURES = len(REDUCED_FEATURES*7) if is_reduced_data else 2807

    # data
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

    # training
    print("Preparing dataset...")
    split = int(0.8*len(data))
    # train set
    df_train = data.iloc[:split].copy()
    print(len(df_train['lumi']), len(df_train['run']))
    print("Training Set")
    print(df_train['lumi'][df_train['run'] == 284044])
    # test set
    df_test = data.iloc[split:].copy()
    print("Testing Set")
    print(df_test['lumi'][df_test['run'] == 284044])