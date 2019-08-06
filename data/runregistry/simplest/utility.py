import os
import pandas as pd

import data.runregistry.simplest.setting as implest_rr_setting

def read_data(selected_pd, pd_data_direcotry=implest_rr_setting.RR_DATA_DIRECTORY):
    return (
        pd.read_csv(os.path.join(pd_data_direcotry, 'good', "{}.csv".format(selected_pd))),
        pd.read_csv(os.path.join(pd_data_direcotry, 'bad', "{}.csv".format(selected_pd)))
    )

def split_dataset(x, frac_test=implest_rr_setting.FRAC_TEST, frac_valid=implest_rr_setting.FRAC_VALID):
    return (
        x.iloc[:-int((frac_test+frac_valid)*len(x)), :].to_numpy() ,
        x.iloc[-int((frac_test+frac_valid)*len(x)):-int(frac_test*len(x)), :].to_numpy() ,
        x.iloc[-int(frac_test*len(x)):, :].to_numpy()
    )
