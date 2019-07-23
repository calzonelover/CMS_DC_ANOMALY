import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data.runregistry.simplest.setting as implest_rr_setting
import data.runregistry.simplest.utility as implest_rr_utility
import data.new_prompt_reco.setting as new_prompt_reco_setting
import data.new_prompt_reco.utility as new_prompt_reco_utility


def main():
    for selected_pd in ["ZeroBias", "JetHT", "EGamma", "SingleMuon"]: 
        print("\n ____ {} ____ channel \n".format(selected_pd))
        good_df, bad_df = implest_rr_utility.read_data(selected_pd=selected_pd)
        n_good, n_bad, n_hcal_bad = good_df.shape[0], bad_df.shape[0], bad_df.query('hcal == 0').shape[0]
        print("# GOOD LS {}, # BAD LS {}".format(n_good, n_bad))
        print("Fraction of HCAL in bad LS {0:.2f}% (# {1}) \n".format( 100.0 * n_hcal_bad/n_bad, n_bad))
        print()
        # visualize
