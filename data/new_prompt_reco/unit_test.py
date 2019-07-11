import numpy as np
import pandas as pd
import os

# customize
from data.new_prompt_reco.setting import (EXTENDED_FEATURES, FEATURES, PDs,
                                            GOOD_DATA_DIRECTORY, PD_GOOD_DATA_DIRECTORY,
                                            BAD_DATA_DIRECTORY, PD_BAD_DATA_DIRECTORY )
import data.new_prompt_reco.utility as utility

def main():
    # settings
    selected_pd = "JetHT"
    features = FEATURES[selected_pd]

    df_bad = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY)
    print(df_bad[['runId', 'lumiId']])
    # utility.extract_and_merge_to_csv(selected_pd, features,
    #                                 data_directory=BAD_DATA_DIRECTORY,
    #                                 pd_data_directory=PD_BAD_DATA_DIRECTORY)

    ## Testing columns name
    # np_dat = np.load("/afs/cern.ch/work/p/ppayoung/public/data2018/golden_json/SingleMuon/crab_20190624_142432/190624_122436/0000/AODTree_40.npy",
    #             encoding="latin1")
    # read_df = pd.DataFrame(np_dat)
    # print(read_df[[*features]])
    # with open('/afs/cern.ch/work/p/ppayoung/public/CMS_DC_ANOMALY/list.txt', 'w') as f:
    #     for feature in read_df.columns:
    #         f.write("{}\n".format(feature))