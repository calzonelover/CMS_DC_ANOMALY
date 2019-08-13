import os
import re, json
import sqlite3
import numpy as np
import pandas as pd

import runregistry as rr

from data.express_2017 import setting as express_2017_settings

dataset_name_cache = {}
lumisection_ranges_cache = {}

class Error(Exception):
    """Base class for other exceptions"""
    pass
class LSNotInRangeError(Error):
    pass

class SubDetector:
    def __init__(self, sub_detector_status):
        self.sub_detector_status = sub_detector_status
        self.n_extended_binsx = int(json.loads(self.sub_detector_status[4])['fXaxis']['fNbins']+2)
        self.n_extended_grids = int(len(json.loads(self.sub_detector_status[4])['fArray']))
        self.extended_data_vec = np.array(json.loads(self.sub_detector_status[4])['fArray'])
    def get_n_binsx(self):
        return int(self.n_extended_binsx-2)
    def get_n_binsy(self):
        return int(self.n_extended_grids/self.n_extended_binsx-2)
    def get_occupancy(self):
        extended_data_grid = np.reshape(self.extended_data_vec, [int(self.n_extended_grids/self.n_extended_binsx), self.n_extended_binsx])
        return extended_data_grid[1:-1,1:-1]

def get_dataset_name(run_id, keyword=express_2017_settings.KEYWORD_FOR_PULLED_LABEL):
    dataset_informations = rr.get_datasets(
        filter={
            'run_number': run_id
        }
    )
    if run_id not in dataset_name_cache:
        try:
            dataset_name_cache[run_id] = tuple(
                map(lambda x: x['name'],
                    filter(lambda x: express_2017_settings.KEYWORD_FOR_PULLED_LABEL in x['name'], dataset_informations)
            ))[0]
            print(run_id, dataset_name_cache[run_id])
        except IndexError:
            print("!! There is no dataset_name for selected keywork '{}' !!".format(express_2017_settings.KEYWORD_FOR_PULLED_LABEL))
            exit()
    return dataset_name_cache[run_id]

def get_lumisection_ranges(run_id, dataset_name):
    lumisection_range_cache_key = "{}___{}".format(dataset_name, run_id)
    if lumisection_range_cache_key not in lumisection_ranges_cache:
        lumisection_ranges_cache[lumisection_range_cache_key] = rr.get_lumisection_ranges(run_id, dataset_name)
    return lumisection_ranges_cache[lumisection_range_cache_key]

def match_lumi_ranges(lumi_id, detector_status_ranges):
    range_lumis = [{'start': int(x['start']), 'end': int(x['end'])} for x in detector_status_ranges]
    if lumi_id > range_lumis[-1]['end'] and lumi_id < range_lumis[0]['start']:
        raise LSNotInRangeError
    for index_range in range(len(range_lumis)):
        if lumi_id >= range_lumis[index_range]['start'] and lumi_id <= range_lumis[index_range]['end']:
            return detector_status_ranges[index_range]    

def main(
        interested_statuses = {
            'hcal_hcal': 'hcal-hcal',
            'ecal_ecal': 'ecal-ecal',
            'tracker_track': 'tracker-track',
            'muon_muon': 'muon-muon'
        },
    ):
    conn = sqlite3.connect(os.path.join(express_2017_settings.RAW_DATA_DIRECTORY, express_2017_settings.SQLITE_RAW_DATA_NAME))
    c = conn.cursor()

    c.execute('SELECT fromrun, torun, fromlumi, tolumi, value, name FROM monitorelements WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"')
    table_rows = c.fetchall()
    table = np.array(table_rows)

    if not np.array_equal(table[:, 0], table[:, 1]) or not np.array_equal(table[:, 2], table[:, 3]):
        print("From {Run,Lumi} to {Run,Lumi} does not correspond!")
        exit()
    run_lumis = tuple(map(lambda row: {'runId': row[0], 'lumiId':row[2]}, table_rows))

    write_data = []
    print(len(run_lumis), len(set(map(lambda run_lumi: run_lumi['runId'], run_lumis))))
    for run_lumi in run_lumis:
        run_id, lumi_id = int(run_lumi['runId']), int(run_lumi['lumiId'])
        c.execute('SELECT fromrun, torun, fromlumi, tolumi, value, name FROM monitorelements WHERE fromrun = {} AND fromlumi = {}'.format(
            run_lumi['runId'], run_lumi['lumiId']
        ))
        table_row = c.fetchall()

        sub_detector_statuses = {
            'runId': run_id,
            'lumiId': lumi_id,
        }
        
        dataset_name = get_dataset_name(run_id=run_id)
        detector_status_ranges = get_lumisection_ranges(run_id=run_id, dataset_name=dataset_name)
        detector_status_range = match_lumi_ranges(lumi_id=lumi_id, detector_status_ranges=detector_status_ranges)
        for sub_detector, sub_detector_str in interested_statuses.items():
            sub_detector_statuses[sub_detector] = detector_status_range[sub_detector_str]
        
        for sub_detector_name in express_2017_settings.CMS_SUBDETECTORS: # ? 63 sub-detector statuses
            try:
                selected_sub_detector_detail = tuple(filter(lambda x: x[5] == sub_detector_name, table_row))[0]
            except IndexError:
                print(sub_detector_name, tuple(filter(lambda x: x[5] == sub_detector_name, table_row)))
            instance_sub_detector = SubDetector(sub_detector_status=selected_sub_detector_detail)
            sub_detector_statuses[sub_detector_name] = instance_sub_detector.get_occupancy()
            
        write_data.append(sub_detector_statuses)

    write_df = pd.DataFrame(write_data)
    write_df.to_csv(os.path.join(express_2017_settings.CLEANED_DATA_DIRECTORY, "occupancy.csv"))