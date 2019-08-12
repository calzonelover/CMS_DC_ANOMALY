# Semi-supervised outlier detection for CMS detector

## Data Quality Monitoring (DQM)
Data quality in CMS detector could be monitored via GUI by using the tools that provided by DQM team. The pipeline of data flow start at online section which shifters at P5 monitor various measured quantity and control the alarming system to directly contact the experts if there is any sub-system went wrong or report the weird behaviour. Not only the online world, DQM tools also provide the offline inspection after 48hrs of collisions to double check the failure of some sub-system by looking multiple histograms.
<p align="center">
    <img src="static/img/dqm_flow.png" width="500px" >
    <br>
    <em>Tools and Processes of DQM, retrieved from M. Schneider, CHEP 2018</em>
</p>

## Data Granularity in CMS (Offline)
* Reconstruction of physics quantities initiate after 48 hours after collisions
* Offline shiters and detector experts check the dozens of distribution histograms to define goodness of data
* Certification is made on **Run and Lumisection levels**
* Lumisection(LS) is taken around 23 seconds

Ref. [1]

## Criteria for bad LS
1) Runs tagged as bad by human (whole run)
2) Automatically filter by DCS bits, beam status and etc. (LS levels)
3) In rare cases are marked by DC experts (LS levels)

The Golden JSON contains the list of all good LS

## Objective
* **Certify data quality in lumisection granularity**
* Reduce manual work of DC Experts

## Expectation
The key concept of this work is to find a decision value to find the cutoff which will be use for certify data quality in LS granularity
<p align="center">
    <img src="static/img/expected_greyzone.png" width="500px" >
    <br>
    <em>Three possible regions of prediction</em>
</p>

## Proposal for An Alternative approach: two steps
* The automatic DCS bit flagging will stay, ML applied on top of it
* Automatize the Data Certification procedure in two steps
  1) Provide a reliable quality flag per Run using grey-zone approach and Supervised models (artificial BAD data can be used for training)
  2) Use Autoencoders only on the grey-zone with the goal to search for anomalous LS and flag them automatically, human double check at this stage
* Using physical quantities as
  * **features** (pT, eta, etc) and
  * **objects** mapped to the relevant Primary Dataset (i.e tracks to
ZeroBias, muons to SingleMuon ... etc)

  to better mimic the current DC procedure
* This repository will cover only on the second step

<p align="center">
    <img src="static/img/cartoon.png" width="700px" >
    <br>
    <em>Pipeline of data certificaiton that we proposed in this work</em>
</p>

Ref. [2]

## References
1) M. Stankevicius, Data Quality Monitoring: Offline
2) F. Fiori, ML Applied To Data CertificationStatus and Perspective

## Dependency
All those script has been done by python3.6 and the libraly version are listed in "requirement.txt"