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
    data["label"] = data.apply(utility.add_flags, axis=1)
    print(data.shape)