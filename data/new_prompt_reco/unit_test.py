import numpy as np
import pandas as pd
import os

# customize
from data.new_prompt_reco.setting import (  EXTENDED_FEATURES, FEATURE_SET_NUMBER, FEATURES, PDs,
                                            GOOD_DATA_DIRECTORY, PD_GOOD_DATA_DIRECTORY,
                                            BAD_DATA_DIRECTORY, PD_BAD_DATA_DIRECTORY,
                                            BAD_DCS_DATA_DIRECTORY, PD_DCS_BAD_DATA_DIRECTORY,
                                            FAILURE_DATA_DIRECTORY, PD_FAILURE_DATA_DIRECTORY, )
import data.new_prompt_reco.utility as utility

def main(
        selected_pd = "JetHT",
        include_failure = False
    ):
    print("//////////////////////////\n     {}     \n//////////////////////////\n".format(selected_pd))
    # features = FEATURES[selected_pd]
    # df = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY, cutoff_eventlumi=False)
    # df_cut = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_GOOD_DATA_DIRECTORY, cutoff_eventlumi=True)
    # print("# good LS: {}, # good LS filter < 500: {}, # filtered_out: {}".format(df.shape[0], df_cut.shape[0], df.shape[0] - df_cut.shape[0]))
    # df = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY, cutoff_eventlumi=False)
    # df_cut = utility.read_data(selected_pd=selected_pd, pd_data_directory=PD_BAD_DATA_DIRECTORY, cutoff_eventlumi=True)
    # print("# bad human LS: {}, # bad human LS filter < 500: {}, # filtered_out: {}".format(df.shape[0], df_cut.shape[0], df.shape[0] - df_cut.shape[0]))

    ## Extract data
    features = FEATURES[selected_pd]

    # utility.extract_and_merge_to_csv(selected_pd, features,
    #                                 data_directory=GOOD_DATA_DIRECTORY,
    #                                 pd_data_directory=PD_GOOD_DATA_DIRECTORY)
    # utility.extract_and_merge_to_csv(selected_pd, features,
    #                                 data_directory=BAD_DATA_DIRECTORY,
    #                                 pd_data_directory=PD_BAD_DATA_DIRECTORY)
    # utility.extract_and_merge_to_csv(selected_pd, features,
    #                                 data_directory=BAD_DCS_DATA_DIRECTORY,
    #                                 pd_data_directory=PD_DCS_BAD_DATA_DIRECTORY)
    if include_failure:
        utility.extract_and_merge_to_csv(
            selected_pd, features,
            data_directory=FAILURE_DATA_DIRECTORY,
            pd_data_directory=PD_FAILURE_DATA_DIRECTORY,
            failure=True
        )

    ## Testing columns name
    # np_dat = np.load("/afs/cern.ch/work/p/ppayoung/public/data2018/golden_json/SingleMuon/crab_20190624_142432/190624_122436/0000/AODTree_40.npy",
    #             encoding="latin1")
    # read_df = pd.DataFrame(np_dat)
    # print(read_df[[*features]])
    # with open('/afs/cern.ch/work/p/ppayoung/public/CMS_DC_ANOMALY/list.txt', 'w') as f:
    #     for feature in read_df.columns:
    #         f.write("{}\n".format(feature))