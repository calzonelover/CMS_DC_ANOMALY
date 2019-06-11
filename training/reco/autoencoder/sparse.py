import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from data.prompt_reco.setting import FEATURES, SELECT_PD
import data.prompt_reco.utility as utility
from model.reco.autoencoder import SparseAutoencoder

def main():
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
    # model
    sparseAutoencoder = SparseAutoencoder(
        summary_dir = "model/reco/summary",
        model_name = "sparse model"
    )
    # test
    print(data['run'].shape)
    print(data['lumi'].shape)
    print(data['label'].shape)
    print(data['label'][20:40])
