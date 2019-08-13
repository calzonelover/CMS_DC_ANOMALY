
KEYWORD_FOR_PULLED_LABEL = "/ReReco/Run2017F_EOY"
SQLITE_RAW_DATA_NAME = "ul2017pilot.sqlite"
RAW_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/express_2017/raw_data/"
CLEANED_DATA_DIRECTORY = "/afs/cern.ch/work/p/ppayoung/public/data2018/express_2017/dataset/"

CMS_SUBDETECTORS = {
  # TRACKER
  "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1",
  "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_2",
  "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_3",
  "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_4",
  "PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_1",
  "PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_2",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_1/TkHMap_NumberValidHits_TECM_W1",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_2/TkHMap_NumberValidHits_TECM_W2",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_3/TkHMap_NumberValidHits_TECM_W3",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_4/TkHMap_NumberValidHits_TECM_W4",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_5/TkHMap_NumberValidHits_TECM_W5",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_6/TkHMap_NumberValidHits_TECM_W6",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_7/TkHMap_NumberValidHits_TECM_W7",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_8/TkHMap_NumberValidHits_TECM_W8",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_9/TkHMap_NumberValidHits_TECM_W9",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_1/TkHMap_NumberValidHits_TECP_W1",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_2/TkHMap_NumberValidHits_TECP_W2",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_3/TkHMap_NumberValidHits_TECP_W3",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_4/TkHMap_NumberValidHits_TECP_W4",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_5/TkHMap_NumberValidHits_TECP_W5",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_6/TkHMap_NumberValidHits_TECP_W6",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_7/TkHMap_NumberValidHits_TECP_W7",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_8/TkHMap_NumberValidHits_TECP_W8",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_9/TkHMap_NumberValidHits_TECP_W9",
  "SiStrip/MechanicalView/TIB/layer_1/TkHMap_NumberValidHits_TIB_L1",
  "SiStrip/MechanicalView/TIB/layer_2/TkHMap_NumberValidHits_TIB_L2",
  "SiStrip/MechanicalView/TIB/layer_3/TkHMap_NumberValidHits_TIB_L3",
  "SiStrip/MechanicalView/TIB/layer_4/TkHMap_NumberValidHits_TIB_L4",
  "SiStrip/MechanicalView/TID/MINUS/wheel_1/TkHMap_NumberValidHits_TIDM_D1",
  "SiStrip/MechanicalView/TID/MINUS/wheel_2/TkHMap_NumberValidHits_TIDM_D2",
  "SiStrip/MechanicalView/TID/MINUS/wheel_3/TkHMap_NumberValidHits_TIDM_D3",
  "SiStrip/MechanicalView/TID/PLUS/wheel_1/TkHMap_NumberValidHits_TIDP_D1",
  "SiStrip/MechanicalView/TID/PLUS/wheel_2/TkHMap_NumberValidHits_TIDP_D2",
  "SiStrip/MechanicalView/TID/PLUS/wheel_3/TkHMap_NumberValidHits_TIDP_D3",
  "SiStrip/MechanicalView/TOB/layer_1/TkHMap_NumberValidHits_TOB_L1",
  "SiStrip/MechanicalView/TOB/layer_2/TkHMap_NumberValidHits_TOB_L2",
  "SiStrip/MechanicalView/TOB/layer_3/TkHMap_NumberValidHits_TOB_L3",
  "SiStrip/MechanicalView/TOB/layer_4/TkHMap_NumberValidHits_TOB_L4",
  "SiStrip/MechanicalView/TOB/layer_5/TkHMap_NumberValidHits_TOB_L5",
  "SiStrip/MechanicalView/TOB/layer_6/TkHMap_NumberValidHits_TOB_L6",

  # ECAL
  # "EcalBarrel/EBOccupancyTask/EBOT digi occupancy",
  # "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE -",
  # "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE +",
  "EcalPreshower/ESOccupancyTask/ES Energy Density Z -1 P 1",
  "EcalPreshower/ESOccupancyTask/ES Energy Density Z -1 P 2",
  "EcalPreshower/ESOccupancyTask/ES Energy Density Z 1 P 1",
  "EcalPreshower/ESOccupancyTask/ES Energy Density Z 1 P 2",

  # "Hcal/DigiRunHarvesting/Occupancy/depth/depth1",
  # "Hcal/DigiRunHarvesting/Occupancy/depth/depth2",
  # "Hcal/DigiRunHarvesting/Occupancy/depth/depth3",
  # "Hcal/DigiRunHarvesting/Occupancy/depth/depth4",

  # Muon
  "CSC/CSCOfflineMonitor/Occupancy/hOStripsAndWiresAndCLCT",
  "RPC/AllHits/SummaryHistograms/Occupancy_for_Barrel",
  "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap",
  # "DT/02-Segments/Wheel-1/numberOfSegments_W-1",
  # "DT/02-Segments/Wheel-2/numberOfSegments_W-2",
  # "DT/02-Segments/Wheel0/numberOfSegments_W0",
  # "DT/02-Segments/Wheel1/numberOfSegments_W1",
  # "DT/02-Segments/Wheel2/numberOfSegments_W2",

  # L1T
  "L1T/L1TObjects/L1TEGamma/timing/egamma_eta_phi_bx_0",
  "L1T/L1TObjects/L1TJet/timing/jet_eta_phi_bx_0",
  "L1T/L1TObjects/L1TMuon/timing/muons_eta_phi_bx_0",
  "L1T/L1TObjects/L1TTau/timing/tau_eta_phi_bx_0",
  "L1T/L1TObjects/L1TEGamma/timing/denominator_egamma",
  "L1T/L1TObjects/L1TJet/timing/denominator_jet",
  "L1T/L1TObjects/L1TMuon/timing/denominator_muons",
  "L1T/L1TObjects/L1TTau/timing/denominator_tau",
}