import numpy as np
import pandas as pd
import os
from data.new_prompt_reco.setting import EXTENDED_FEATURES, FEATURES, PDs, GOOD_DATA_DIRECTORY, PD_GOOD_DATA_DIRECTORY

def extract_and_merge_to_csv(selected_pd, features, good_data_directory=GOOD_DATA_DIRECTORY, pd_good_data_directory=PD_GOOD_DATA_DIRECTORY):
    list_df = []
    for crab in os.listdir(os.path.join(good_data_directory, selected_pd)):
        print(crab)
        for output in os.listdir(os.path.join(good_data_directory, selected_pd, crab)):
            print(" -", output)
            for run in os.listdir(os.path.join(good_data_directory, selected_pd, crab, output)):
                print("  *", run)
                for dat_numpy in os.listdir(os.path.join(good_data_directory, selected_pd, crab, output, run)):
                    print("   ~", dat_numpy)
                    write_df = pd.DataFrame()
                    np_dat = np.load(os.path.join(good_data_directory, selected_pd, crab, output, run, dat_numpy),
                                    encoding="latin1")
                    read_df = pd.DataFrame(np_dat)
                    for feature in features:
                        tags = read_df[feature].apply(pd.Series)
                        tags = tags.rename(columns = lambda x : '{}_'.format(feature) + str(x))
                        read_df.drop(columns=feature)
                        for i in range(0,7):
                            write_df['{}_{}'.format(feature, i)] = tags['{}_{}'.format(feature, i)]
                    for feature in EXTENDED_FEATURES:
                        write_df[feature] = read_df[feature]
                    list_df.append(write_df)
    full_df = pd.concat(list_df)
    print(full_df)
    if not os.path.exists(pd_good_data_directory):
        os.mkdir(pd_good_data_directory)
    full_df.to_csv(os.path.join(pd_good_data_directory, "{}.csv".format(selected_pd)))