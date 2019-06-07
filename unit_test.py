import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from data.prompt_reco.setting import FEATURES
import data.prompt_reco.utility as utility

def read_data():
    files = utility.get_file_list(chosed_pd=21)

    feature_names = utility.get_feature_name(FEATURES)
    data = pd.DataFrame(get_data(files), columns=feature_names)

if __name__ == "__main__":
    read_data()