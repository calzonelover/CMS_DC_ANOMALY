import numpy as np
import pandas as pd
import os
import re
from data.new_prompt_reco import setting
from data.new_prompt_reco import utility

import runregistry as rr

class Error(Exception):
    """Base class for other exceptions"""
    pass
class PromptRecoDatasetNotUniqueError(Error):
    pass
class LSNotInRangeError(Error):
    pass

regex_cache = {}
dataset_name_cache = {}
lumisection_ranges_cache = {}

def get_dataset_name(recon_name="PromptReco", run_id=None):
    dataset_name_cache_key = "%s___%s" % (recon_name, run_id)
    if dataset_name_cache_key in dataset_name_cache:
        # print("Yay, found %s" % (dataset_name_cache_key))
        return dataset_name_cache[dataset_name_cache_key]

    if recon_name in regex_cache:
        r = regex_cache[recon_name]
    else:
        r = re.compile(".*{}".format(recon_name))
        regex_cache[recon_name] = r

    dataset_informations = rr.get_datasets(
        filter={
            'run_number': int(run_id)
        }
    )
    dataset_names = list(map(lambda x: x['name'], dataset_informations))
    prompt_reco_dataset = list(filter(r.match, dataset_names))
    if len(prompt_reco_dataset) > 1:
        raise PromptRecoDatasetNotUniqueError

    dataset_name_cache[dataset_name_cache_key] = prompt_reco_dataset[0]
    return prompt_reco_dataset[0]


def get_lumisection_ranges(run_id, prompt_reco_dataset):
    lumisection_ranges_cache_key = "%s___%s" % (run_id, prompt_reco_dataset)
    if lumisection_ranges_cache_key in lumisection_ranges_cache:
        # print("Double yay, found %s" % (lumisection_ranges_cache_key))
        return lumisection_ranges_cache[lumisection_ranges_cache_key]

    ranges = rr.get_lumisection_ranges(run_id, prompt_reco_dataset)
    lumisection_ranges_cache[lumisection_ranges_cache_key] = ranges
    return ranges

def match_lumi_range(lumi_id, detector_status_ranges):
    range_lumis = [{'start': int(x['start']), 'end': int(x['end'])} for x in detector_status_ranges]
    if lumi_id > range_lumis[-1]['end'] and lumi_id < range_lumis[0]['start']:
        raise LSNotInRangeError
    for index_range in range(len(range_lumis)):
        if lumi_id >= range_lumis[index_range]['start'] and lumi_id <= range_lumis[index_range]['end']:
            return detector_status_ranges[index_range]


def main(
        selected_pd = "JetHT",
        recon_name = "PromptReco",
        interested_statuses = {
            'hcal_hcal': 'hcal-hcal',
            'ecal_ecal': 'ecal-ecal',
            'tracker_track': 'tracker-track',
            'muon_muon': 'muon-muon'
        },
        include_bad_dcs = False,
    ):
    print("\n\n Extract {} dataset \n\n".format(selected_pd))
    df_good = utility.read_data(selected_pd=selected_pd, pd_data_directory=setting.PD_GOOD_DATA_DIRECTORY)
    df_bad_human = utility.read_data(selected_pd=selected_pd, pd_data_directory=setting.PD_BAD_DATA_DIRECTORY)
    df_bad_dcs = utility.read_data(selected_pd=selected_pd, pd_data_directory=setting.PD_DCS_BAD_DATA_DIRECTORY)
    if include_bad_dcs:
        df_bad = pd.concat([df_bad_human, df_bad_dcs], ignore_index=True)
    else:
        df_bad = df_bad_human
    df_write_good = df_good
    for sub_detector, sub_detector_str in interested_statuses.items():
        df_write_good[sub_detector] = 1
    
    sub_detector_statuses = []
    for df in [df_bad, ]:
        for row_i in range(df.shape[0]):
            run_id, lumi_id = int(df['runId'][row_i]), int(df['lumiId'][row_i])
            if not row_i % 1000:
                print("process %.2f%% (%s/%s), run# %s, lumi# %s" % (100.0 * row_i/df.shape[0], row_i, df.shape[0], run_id, lumi_id))

            prompt_reco_dataset = get_dataset_name(recon_name=recon_name, run_id=run_id)
            detector_status_ranges = get_lumisection_ranges(run_id, prompt_reco_dataset)
            detector_status_range = match_lumi_range(lumi_id, detector_status_ranges)

            sub_detector_status = [
                1 if detector_status_range[sub_detector_str]['status'] == 'GOOD' else 0
                for sub_detector, sub_detector_str in interested_statuses.items()
            ]
            if 0 in sub_detector_status: print("Found bad sub-system in run {} LS {}!!".format(run_id, lumi_id),sub_detector_status)
            sub_detector_statuses.append(sub_detector_status)
        df_label = pd.DataFrame(sub_detector_statuses, columns = [ sub_detector for sub_detector, sub_detector_str in interested_statuses.items() ] )
        df_write = pd.concat([df, df_label], axis=1)
        try:
            df_write.to_csv(os.path.join(setting.PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY, 'bad', "{}_feature{}.csv".format(selected_pd, setting.FEATURE_SET_NUMBER)))
        except FileNotFoundError:
            os.mkdir(os.path.join(setting.PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY, 'bad'))
            df_write.to_csv(os.path.join(setting.PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY, 'bad', "{}_feature{}.csv".format(selected_pd, setting.FEATURE_SET_NUMBER)))
    try:
        df_write_good.to_csv(os.path.join(setting.PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY, 'good', "{}_feature{}.csv".format(selected_pd, setting.FEATURE_SET_NUMBER)))
    except FileNotFoundError:
        os.mkdir(os.path.join(setting.PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY, 'good'))
        df_write_good.to_csv(os.path.join(setting.PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY, 'good', "{}_feature{}.csv".format(selected_pd, setting.FEATURE_SET_NUMBER)))