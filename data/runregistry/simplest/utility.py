import os
import pandas as pd

import data.runregistry.simplest.setting as implest_rr_setting

def read_data(selected_pd, pd_data_direcotry=implest_rr_setting.RR_DATA_DIRECTORY):
    return (
        pd.read_csv(os.path.join(pd_data_direcotry, 'good', "{}.csv".format(selected_pd))),
        pd.read_csv(os.path.join(pd_data_direcotry, 'bad', "{}.csv".format(selected_pd)))
    )