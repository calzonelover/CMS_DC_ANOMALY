# Report for Express datasets (offline)

## Subdetector: RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap
Regards for 8x5 grid that contains occupancy of events as the example figure below
<p align="center">
<img src="occupancy_endcap_ex.png" width="800px" >
</p>

### K-Means
with fixing N_CLASSES = 2 under assumption of good and poor conditions
 * non-preprocess data
<p align="center">
<img src="KMeans/KMeans_Clustering.png" width="500px" >
</p>

 * Standardized data (shifted mean and divide by variance by column)
<p align="center">
<img src="KMeans/KMeans_Clustering_with_StandardScalar.png" width="500px" >
</p>

 * Normalized data (by row)
<p align="center">
<img src="KMeans/KMeans_Clustering_normalize_row.png" width="500px" >
</p>