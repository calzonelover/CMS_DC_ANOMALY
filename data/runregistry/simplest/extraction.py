import os
import re, json
import pandas as pd
import numpy as np

import runregistry as rr

import data.runregistry.simplest.setting as implest_rr_setting
import data.new_prompt_reco.setting as new_prompt_reco_setting
import data.new_prompt_reco.utility as new_prompt_reco_utility

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

def main(
        selected_pd = "JetHT",
        recon_name = "PromptReco",
        interested_statuses = {
            'hcal_hcal': 'hcal-hcal',
            'ecal_ecal': 'ecal-ecal',
            'tracker_track': 'tracker-track',
            'muon_muon': 'muon-muon'
        },
    ):
    print("\n\n Extract {} dataset \n\n".format(selected_pd))
    features = new_prompt_reco_utility.get_full_features(selected_pd)
    df_good = new_prompt_reco_utility.read_data(selected_pd=selected_pd, pd_data_directory=new_prompt_reco_setting.PD_GOOD_DATA_DIRECTORY)
    df_bad = new_prompt_reco_utility.read_data(selected_pd=selected_pd, pd_data_directory=new_prompt_reco_setting.PD_BAD_DATA_DIRECTORY)
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

            range_lumis = [{'start': int(x['start']), 'end': int(x['end'])} for x in detector_status_ranges]
            if lumi_id > range_lumis[-1]['end'] and lumi_id < range_lumis[0]['start']:
                raise LSNotInRangeError

            for index_range in range(len(range_lumis)):
                if lumi_id >= range_lumis[index_range]['start'] and lumi_id <= range_lumis[index_range]['end']:
                    detector_status_range = detector_status_ranges[index_range]
                    ##
                    # print(detector_status_range.keys())
                    # input()
                    ##
                    sub_detector_status = [
                        1 if detector_status_range[sub_detector_str]['status'] == 'GOOD' else 0
                        for sub_detector, sub_detector_str in interested_statuses.items()
                    ]
                    sub_detector_statuses.append(sub_detector_status)
                        
                    # hb_status = 1 if detector_status_range['hcal-hb']['status'] == 'GOOD' else 0
                    # he_status = 1 if detector_status_range['hcal-he']['status'] == 'GOOD' else 0
                    # hf_status = 1 if detector_status_range['hcal-hf']['status'] == 'GOOD' else 0
                    # h_status = 1 if detector_status_range['hcal-hcal']['status'] == 'GOOD' else 0
                    # h_all_status = hb_status * he_status * hf_status * h_status
                    # e_status = 1 if detector_status_range['ecal-ecal']['status'] == 'GOOD' else 0
                    # sub_detector_statuses.append([hb_status, he_status, hf_status, h_status, h_all_status, e_status])
                    if 0 in sub_detector_status:
                        print(
                            "Found bad HCAL in run {} LS {}!!".format(run_id, lumi_id),
                            sub_detector_status
                        )
    df_label = pd.DataFrame(sub_detector_statuses, columns = [ sub_detector for sub_detector, sub_detector_str in interested_statuses.items() ] )
    df_write_bad = pd.concat([df_bad, df_label], axis=1)
    df_write_good.to_csv(os.path.join(implest_rr_setting.RR_DATA_DIRECTORY, 'good', "{}.csv".format(selected_pd)))
    df_write_bad.to_csv(os.path.join(implest_rr_setting.RR_DATA_DIRECTORY, 'bad', "{}.csv".format(selected_pd)))

    # 1) get data_name from RR by runID
    # 2) get prompt_reco from those