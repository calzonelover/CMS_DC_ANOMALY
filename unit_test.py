# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from data.prompt_reco.setting import FEATURES
import data.prompt_reco.utility as utility

def read_data():
    files = utility.get_file_list(chosed_pd=21)

    feature_names = utility.get_feature_name()

    data = pd.DataFrame(utility.get_data(files), columns=feature_names)
    data["run"] = data["run"].astype(int)
    data["lumi"] = data["lumi"].astype(int)
    data.drop(["_foo", "_bar", "_baz"], axis=1, inplace=True)
    data = data.sort_values(["run", "lumi"], ascending=[True,True])
    data = data.reset_index(drop=True)
    print(data['run'].shape)
    print(data['lumi'].shape)
    print(data[['run', 'lumi', 'qNVtx_1','qNVtx_2','qNVtx_3','qNVtx_4','qNVtx_5','qNVtx_6','qNVtx_7']])

if __name__ == "__main__":
    read_data()