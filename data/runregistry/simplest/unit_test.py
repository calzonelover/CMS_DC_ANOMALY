import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data.runregistry.simplest.setting as implest_rr_setting
import data.runregistry.simplest.utility as implest_rr_utility
import data.new_prompt_reco.setting as new_prompt_reco_setting
import data.new_prompt_reco.utility as new_prompt_reco_utility


def get_fraction(
        PDs = ["ZeroBias", "JetHT", "EGamma", "SingleMuon"],
        interested_statuses = {
            'hcal_hcal': 'hcal-hcal',
            'ecal_ecal': 'ecal-ecal',
            'tracker_track': 'tracker-track',
            'muon_muon': 'muon-muon'
        },
    ):
    for selected_pd in PDs:
        print("\n ____ {} ____ channel \n".format(selected_pd))
        good_df, bad_df = implest_rr_utility.read_data(selected_pd=selected_pd)
        n_good, n_bad = good_df.shape[0], bad_df.shape[0]
        print("# GOOD LS {}, # BAD LS {}".format(n_good, n_bad))
        sum_percent = 0.0
        for sub_detector, sub_detector_str in interested_statuses.items():
                n_specific_bad = bad_df.query('{} == 0'.format(sub_detector)).shape[0]
                print("Fraction of {0} in bad LS {1:.2f}% (# {2}) \n".format(sub_detector, 100.0 * n_specific_bad/n_bad, n_specific_bad))
                sum_percent += 100.0 * n_specific_bad/n_bad
        print("Other cases {0:.2f}%: negative if there is more then one sub-system malfunction at the same time".format(100.0 - sum_percent))
        # visualize