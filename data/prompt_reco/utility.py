import numpy as np
import h5py
from data.prompt_reco.setting import DATA_DIRECTORY, PDs, FEATURES, PERIOD_DATA_TAKINGs, EXTENDED_FEATURES

def get_feature_name(features=FEATURES, extended_features=EXTENDED_FEATURES):
    feature_names = []
    for feature in features:
        for i in range(1, 8):
            feature_names.append("%s_%s" % (feature, i))
    feature_names.extend(extended_features)
    return feature_names

def get_file_list(chosed_pd, data_dir=DATA_DIRECTORY, pds=PDs, period_data_takings=PERIOD_DATA_TAKINGs, type_dats=['signal', 'background'], extension='h5'):
    '''
        data_dir: 
        pds: primary datasets
        chosed_pd: # of chosen pd
        type_dat: typically signal, background
    '''
    files = []
    for type_dat in type_dats:
        for period in period_data_takings:
            files.append("{}{}_{}_{}.{}".format(data_dir, pds[chosed_pd], period, type_dat, extension))
    return files

def get_data(file_dats):
    for file_dat in file_dats:        
        jet = file_dat.split("/")[-1][:-3]
        print(jet)
        try:
            h5file = h5py.File(file_dat, "r")
            print(h5file[jet].shape)
            readout = np.concatenate((np.empty([0,2813]), h5file[jet]), axis=0)
        except OSError as error:
            print("This Primary Dataset doesn't have %s. %s" % (jet, error))
            continue
    return readout