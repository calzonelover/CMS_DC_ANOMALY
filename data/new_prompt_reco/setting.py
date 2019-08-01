from data.new_prompt_reco.features import (ZEROBIAS_FEATURES, JETHT_FEATURES, EGAMMA_FEATURES, SINGLEMUON_FEATURES)

FRAC_VALID, FRAC_TEST = 0.2, 0.2

PD_GOOD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/prompt_reco_2018/good_data/"
PD_BAD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/prompt_reco_2018/bad_data/"
PD_FAILURE_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/raw_prompt_reco_2018/failures/"

GOOD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/raw_prompt_reco_2018/golden_json/"
BAD_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/raw_prompt_reco_2018/bad_json/"
FAILURE_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/raw_prompt_reco_2018/failures/"

PDs = {
    1: 'ZeroBias',
    2: 'JetHT',
    3: 'EGamma',
    4: 'SingleMuon'
}

EXTENDED_FEATURES = ["runId", "lumiId", "lumi"]

FEATURES = {
    'ZeroBias': ZEROBIAS_FEATURES,
    'JetHT': JETHT_FEATURES,
    'EGamma': EGAMMA_FEATURES,
    'SingleMuon': SINGLEMUON_FEATURES,
}