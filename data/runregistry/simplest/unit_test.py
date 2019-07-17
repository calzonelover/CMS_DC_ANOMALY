import pandas as pd
import numpy as np
import json

import runregistry as rr

import data.runregistry.simplest.setting as implest_rr_setting

import data.new_prompt_reco.setting as new_prompt_reco_setting
import data.new_prompt_reco.utility as new_prompt_reco_utility

def main(
        selected_pd = "JetHT",
    ):

    features = new_prompt_reco_utility.get_full_features(selected_pd)
    df_good = new_prompt_reco_utility.read_data(selected_pd=selected_pd, pd_data_directory=new_prompt_reco_setting.PD_GOOD_DATA_DIRECTORY)
    df_bad = new_prompt_reco_utility.read_data(selected_pd=selected_pd, pd_data_directory=new_prompt_reco_setting.PD_BAD_DATA_DIRECTORY)

    runID_good, lumiID_good = df_good['runId'], df_good['lumiId']
    runID_bad, lumiID_bad = df_bad['runId'], df_bad['lumiId']

    write_df_HCAL = pd.DataFrame()
    for runIDs, lumiIDs in zip([runID_good, runID_bad], [lumiID_good, lumiID_bad]):
        for run_id in runIDs.to_numpy():
            print("run id : {}".format(int(run_id)))
            dataset_names = rr.get_datasets(
                filter={
                    'run_number': int(run_id)
                }
            )
            print(json.dumps(dataset_names, indent=4, sort_keys=True))
            # print(list(map(lambda x: x['class'], dataset_names)))

            # print("Dataset name: {}".format(dataset_names)

            # subdetector_statuses = rr.get_lumisections(int(run_id), "/PromptReco/Collisions2018D/DQM")
            # print("# sub-detector status: {}, # lumi {}".format(len(subdetector_statuses), len(lumiIDs)))

            # shifted_lumiIDs = list(map(lambda x: x-1, lumiIDs.to_numpy()))
            # hcal_statuses = [ subdetector_statuses[shifted_lumiID] for shifted_lumiID in shifted_lumiIDs ]


            # 1) get data_name from RR by runID
            # 2) get prompt_reco from those 
