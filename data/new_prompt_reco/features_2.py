#####################
#      set 2        #
#####################

FIX_FEATURE_COLUMNS = {

}

ZEROBIAS_FEATURES = [
    'qpVtxChi2',
    'qpVtxNtr',
    'qpVtxX',
    'qpVtxY',
    'qpVtxZ',
    'qPUEvt',
    'qlumiEvt',

    'qgTkPt',
    'qgTkEta',
    'qgTkPhi',
    'qgTkN',
    'qgTkChi2',
    'qgTkNHits',
    'qgTkNLay',
]

JETHT_FEATURES = [
    "qpVtxChi2",
    "qpVtxNtr",
    "qpVtxX",
    "qpVtxY",
    "qpVtxZ",
    "qPUEvt",
    "qlumiEvt",

    "qPFJetN",
    "qPFJetPt",
    "qPFJetPhi",
    "qPFJetEta",
    "qPFMetPt",
    "qPFMetPhi",

    "qCalJetN",
    "qCalJetPt",
    "qCalJetEta",
    "qCalJetPhi",
    "qCalJetEn",
    "qCalMETPt",
    "qCalMETPhi",

    "qCCEn", "qCCEta", "qCCPhi",
    "qSCEn", "qSCEta", "qSCPhi",
]

EGAMMA_FEATURES = [
    "qpVtxChi2",
    "qpVtxNtr",
    "qpVtxX_",
    "qpVtxY_",
    "qpVtxZ_",
    "qPUEvt_",
    "qlumiEvt",

    "qGsfPt",
    "qGsfEta",
    "qGsfPhi",
    "qGsfN",

    "qPhoN",
    # "qPhoPt",
    # "qPhoEta",
    # "qPhoPhi",
    # "qPhoEn",
    # "qPhoe1x5",
    # "qPhoe3x3",

    "qgedPhoPt",
    "qgedPhoEta",
    "qgedPhoPhi",
    "qgedPhoEn_",
    "qgedPhoe1x5_",
    "qgedPhoe3x3_",

    "qSigmaIEta", "qSigmaIPhi", "qr9", "qHadOEm",
    "qdrSumPt", "qdrSumEt", "qeSCOP", "qecEn"
]

SINGLEMUON_FEATURES = [
    "qpVtxChi2_",
    "qpVtxNtr_",
    "qpVtxX_",
    "qpVtxY_",
    "qpVtxZ_",
    "qPUEvt_",
    "qlumiEvt_",

    "qglobTkN",
    "qglobTkPt",
    "qglobTkEta",   
    "qglobTkPhi",
    "qglobTkN_",
    "qglobTkChi2",
    "qglobTkNHits",

    "qMuN",
    "qMuPt",
    "qMuEta",
    "qMuPhi",
    "qMuEn",
    "qMuChi2",
]