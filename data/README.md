# Offline (PromptReco)
* pp collisions, PromptReco (currently 2018 data, but we also provide the tools for access 2016 data)
* Each lumisection (datapoint) contains
  * 39 histogram of physics quantity e.g. JetPt, JetEta, JetPhi, etc.
  * Represent one histogram with 7 numbers
  * 259 Features (39 × 7)
* <span style="color:green">Good LS</span> defined in Golden JSON else <span style="color:red">Bad LS</span>

## Histogram representation
* Collection of physics objects e.g. photons, muons and so on
* Measurement quantity: Transverse momentum, eta, phi,
etc.
* For full detail of selected features could be found in "features_x.py"

<p align="center">
    <img src="../static/img/ex_eta_dist.png" width="600px" >
    <br>
    <em>Example histogram of Eta distribution</em>
</p>

1) Quantize [10%, 30%, 50%, 70%, 90%] of the histogram
2) Combine mean and rms
3) Use these **7 values to
represent one histogram**

## Data preprocessing
* **MinMaxScalar** Transformation
* Consider Lumisection i and Feature j
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?x_{ij}'&space;\leftarrow&space;\frac{x_{ij}&space;-&space;\min_{\forall&space;i\in&space;S_{\text{train}}}\{x_{ij}\}}{\max_{\forall&space;i\in&space;S_{\text{train}}}\{x_{ij}\}&space;-&space;\min_{\forall&space;i\in&space;S_{\text{train}}}\{x_{ij}\}&space;}" title="x_{ij}' \leftarrow \frac{x_{ij} - \min_{\forall i\in S_{\text{train}}}\{x_{ij}\}}{\max_{\forall i\in S_{\text{train}}}\{x_{ij}\} - \min_{\forall i\in S_{\text{train}}}\{x_{ij}\} }" />
</p>

* Then our datapoint should be in range [0, 1]

<!-- # Online (Occupancy) -->