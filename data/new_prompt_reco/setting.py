from data.new_prompt_reco import features, features_2

FEATURE_SET_NUMBER = 2
CUTOFF_VALUE_EVENTS_LUMI = 500
FRAC_VALID, FRAC_TEST = 0.2, 0.2


# peronal storage of patomp
PD_GOOD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/prompt_reco_2018/good_data/"
PD_BAD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/prompt_reco_2018/bad_data/"
PD_DCS_BAD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/prompt_reco_2018/dcs_bad_data/"
PD_FAILURE_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/prompt_reco_2018/failures/"

PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/prompt_reco_2018/pull_labeled_human_dcs_bad/bad/"
PD_LABELED_SUBSYSTEM_GOOD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/prompt_reco_2018/pull_labeled_human_dcs_bad/good/"

'''
# In case afs storage of mine has been removed, you could use the following storage
PD_GOOD_DATA_DIRECTORY = "/eos/cms/store/group/comm_dqm/ML4DC_2019/ML4DC_NUMPY_2018/prompt_reco_2018/good_data/"
PD_BAD_DATA_DIRECTORY = "/eos/cms/store/group/comm_dqm/ML4DC_2019/ML4DC_NUMPY_2018/prompt_reco_2018/bad_data/"
PD_DCS_BAD_DATA_DIRECTORY = "/eos/cms/store/group/comm_dqm/ML4DC_2019/ML4DC_NUMPY_2018/prompt_reco_2018/dcs_bad_data/"
PD_FAILURE_DATA_DIRECTORY = "/eos/cms/store/group/comm_dqm/ML4DC_2019/ML4DC_NUMPY_2018/prompt_reco_2018/failures/"

PD_LABELED_SUBSYSTEM_BAD_DATA_DIRECTORY = "/eos/cms/store/group/comm_dqm/ML4DC_2019/ML4DC_NUMPY_2018/prompt_reco_2018/pull_labeled_human_dcs_bad/bad/"
PD_LABELED_SUBSYSTEM_GOOD_DATA_DIRECTORY = "/eos/cms/store/group/comm_dqm/ML4DC_2019/ML4DC_NUMPY_2018/prompt_reco_2018/pull_labeled_human_dcs_bad/good/"
'''

GOOD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/raw_prompt_reco_2018/good/"
BAD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/raw_prompt_reco_2018/human_bad/"
BAD_DCS_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/raw_prompt_reco_2018/dcs_bad/"
FAILURE_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/raw_prompt_reco_2018/failures/"

PDs = {
    1: 'ZeroBias',
    2: 'JetHT',
    3: 'EGamma',
    4: 'SingleMuon'
}

EXTENDED_FEATURES = ["runId", "lumiId", "lumi", "EventsPerLs"]

FEATURESETS = {
    1: {
        'ZeroBias': features.ZEROBIAS_FEATURES,
        'JetHT': features.JETHT_FEATURES,
        'EGamma': features.EGAMMA_FEATURES,
        'SingleMuon': features.SINGLEMUON_FEATURES,
    },
    2: {
        'ZeroBias': features_2.ZEROBIAS_FEATURES,
        'JetHT': features_2.JETHT_FEATURES,
        'EGamma': features_2.EGAMMA_FEATURES,
        'SingleMuon': features_2.SINGLEMUON_FEATURES,
    },
}
FEATURES = FEATURESETS[FEATURE_SET_NUMBER]

FIX_FEATURE_COLUMNS_SETS = {
    1: features.FIX_FEATURE_COLUMNS,
    2: features_2.FIX_FEATURE_COLUMNS,
}
FIX_FEATURE_COLUMNS = FIX_FEATURE_COLUMNS_SETS[FEATURE_SET_NUMBER]