#####################
#      set 1        #
#####################

FIX_FEATURE_COLUMNS = {
    "qpVtxChi2_": "qpVtxChi2",
    "qpVtxNtr_": "qpVtxNtr",
    "qpVtxX_": "qpVtxX",
    "qpVtxY_": "qpVtxY",
    "qpVtxZ_": "qpVtxZ",
    "qPhoEn_": "qPhoEn",
    "qPhoe1x5_": "qPhoe1x5",
    "qPhoe3x3_": "qPhoe3x3",
    "qgedPhoEn_": "qgedPhoEn",
    "qgedPhoe1x5_": "qgedPhoe1x5",
    "qgedPhoe3x3_": "qgedPhoe3x3",
    "qMuEn_": "qMuEn",
    "qMuChi2_": "qMuChi2",
}

ZEROBIAS_FEATURES = [
    "qpVtxChi2_",
    "qpVtxNtr_",
    "qpVtxX_",
    "qpVtxY_",
    "qpVtxZ_",
    "qPUEvt",
    "qlumiEvt",

    "qgTkPt",
    "qgTkEta",
    "qgTkPhi",
    "qgTkN",
    "qgTkChi2",
    "qgTkNHits",
]

JETHT_FEATURES = [
    "qpVtxChi2_",
    "qpVtxNtr_",
    "qpVtxX_",
    "qpVtxY_",
    "qpVtxZ_",
    "qPUEvt",
    "qlumiEvt",

    "qPFJetPt",
    "qPFJetPhi",
    "qPFJetEta",
    "qPFMetPt",
    "qPFMetPhi",

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
    "qpVtxChi2_",
    "qpVtxNtr_",
    "qpVtxX_",
    "qpVtxY_",
    "qpVtxZ_",
    "qPUEvt",
    "qlumiEvt",

    "qgTkPt",
    "qgTkEta",
    "qgTkPhi",

    "qPhoPt",
    "qPhoEta",
    "qPhoPhi",
    "qPhoEn_",
    "qPhoe1x5_",
    "qPhoe3x3_", 

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
    "qPUEvt",
    "qlumiEvt",

    "qglobTkPt",
    "qglobTkEta",   
    "qglobTkPhi",
    "qglobTkN",
    "qglobTkChi2",
    "qglobTkNHits",

    "qMuPt",
    "qMuEta",
    "qMuPhi",
    "qMuEn_",
    "qMuChi2_",
]