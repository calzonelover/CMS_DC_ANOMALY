# 2016 Dataset (Only JetHT)

### Primary Analysis
In order to roughly understand a group (similar patterns) of data, one way to do it is to reduce the dimension of data. In our case, there are 259 features which will be transformed into two dimension on the basis of two eigenvectors (selected by two largest eigenvalues) belonging to covariance matrix which computed from the datasets. You could checkout more nicer mathematical exprssion in [this link](https://www2.imm.dtu.dk/pubdb/views/edoc_download.php/4000/pdf/imm4000).

<p align="center">
    <img src="Old_Data/logs/ReducedFeatures/minmaxscalar/JetHT_label.png" width="500px" >
    <br>
    <em>Principal component with the labeled color from the system</em>
</p>

As you could see on the green line that there are nice band which is good LS and a few weird LSs that located outside the tubular shape as well as bad LS that could be divided into the bad LS with some patterns and anamaly bad LS which I would called both of them as "outlier". That's essentially the punchline why I called outlier detection instead of anomaly detection.

### Performance
We Iteratively retrain the model ten times to make sure that it's working systematically and plot the root mean square as a shady fluctuation in the following figures

<p align="center">
    <img src="Old_Data/logs/ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle/performance.png" width="500px" >
    <br>
    <em>Various AE</em>
</p>
<p align="center">
    <img src="Old_Data/logs/ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle/performance_ml.png" width="500px" >
    <br>
    <em>Vanilla AE vs OneClass-SVM vs Isolation Forest</em>
</p>

To sum up, even there are fancy mathematical expression of non-vanilla autoencoders but it does not guarantee that we would get the best performance out if it. On the other hand, **simplest AE has the performance** among all AE. One other intersting spot is the performance of **OneClass-SVM also yield the remarkable results** as nearly compatible with Vanilla AE without any fluctuation since the model itself has no randomness and working very strightforward. 

### Distribution of decision value (to find the threshold)
The story behind the performance figure is genuinely extracted from the distribution of decision value and slowly moving a threshold of minimal point in the overlapping region of good and bad LS from label in the distribution until it got the maximal value. The below figures are the comparison between our two great candidates by consider to pick some threshold and see the contamination in each side

<p align="center">
    <img src="Old_Data/logs/ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle/decision_value_dist.png" width="700px" >
</p>

For Vanilla AE, the contamination of bad LS falling into good LS is around 1% over the good LS below the cutoff and there are only countable of good LS falling into bad LS which might be ignorable.

For OneClass-SVM, the contamination LS that bad falling into good LS is almost exactly the same as Vanilla AE does. There is no coincidence for totally different approach of model train and spot the same thing. This might implicitly implies that it either came from some imperfection of data in the training and testing or some kind of malfunction in the sub-system couldn't propagated into JetHT physics objects.

As can be seen in the distribution, there is no clear grey zone for this study so far.

### Example of square error from reconstruction
Here are the example of LS reconstruction which calculated from x and x' between good and bad LS
<p align="center">
    <img src="Old_Data/logs/ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle/example_se.png" width="500px" >
    <br>
    <em>Colorize reconstruction error from Vanilla AE</em>
</p>

### Extended Investigation
You might wondering why many of bad LS seems to have a group of bad LS as you have seen in the plot of hyperspace and few collection of bad LS in decision value distribution (As the black arrow that link between the distribution and 2D-hyperspace). In this section, I want to explicitly prove that the model really see that the right cluster is the worse bad LS and more closer to tubular is less bad LS which decision value have to be quite similar to good LS. In order to prove that, I choose our best candidate to shading the decision value as z-axis color to represent how bad LS in each data point is.
<p align="center">
    <img src="Old_Data/logs/ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle/guess_visual.png" width="700px" >
    <br>
    <em>Reconstruction error from Vanilla AE</em>
</p>

### Summary
* Semi-supervised learning yield a remarkable result and well describe outlier LS
* There is no grey zone from our model for this dataset
* Bad LS could be divided into two parts
  * Bad with some pattern
  * Anomaly

For the weekly report which contain the full detail of this study please checkout [this direcotry](Old_Data/reports/).

# 2018 Dataset

### Primary Analysis
For 2018 data, we dig a bit more to understand which cause the badness of bad LS by taking sub-system label into account from 
[RR's API](https://github.com/fabioespinosa/runregistry_api_client). There are a plenty of sub-system in CMS detector. In order to roughly understand the malfunction of sub-system, we decided to pull label only for HCAL, ECAL, TRACKER and MUON detector which are the main part of the detector. In order to roughly describe each features contribute to each principal components, we extract the element in matrix transform (equivalent to an element in each eigenvector) and take the absolute value to consider only for the magnitude and ignore the direction in the space where it directly proportional to contribution of each one.

* EGamma
  
<p align="center">
    <img src="new_data/logs/minmaxscalar/LastSubmission/EGamma_subsystem_label.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/EGamma_subsystem_label_short_range.png" width="400px" >
</p>

<p align="center">
    <img src="new_data/logs/minmaxscalar/EGamma_pc1.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/EGamma_pc2.png" width="400px" >
</p>


* Single Muon
<p align="center">
    <img src="new_data/logs/minmaxscalar/LastSubmission/SingleMuon_label_separate.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/SingleMuon_label_separate_short_range.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/SingleMuon_subsystem_label.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/SingleMuon_subsystem_label_short_range.png" width="400px" >
</p>

<p align="center">
    <img src="new_data/logs/minmaxscalar/SingleMuon_pc1.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/SingleMuon_pc2.png" width="400px" >
</p>

* ZeroBias
<p align="center">
    <img src="new_data/logs/minmaxscalar/LastSubmission/ZeroBias_label_separate.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/ZeroBias_label_separate_short_range.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/ZeroBias_subsystem_label.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/ZeroBias_subsystem_label_short_range.png" width="400px" >
</p>

<p align="center">
    <img src="new_data/logs/minmaxscalar/ZeroBias_pc1.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/ZeroBias_pc2.png" width="400px" >
</p>

* JetHT
<p align="center">
    <img src="new_data/logs/minmaxscalar/LastSubmission/JetHT_label_separate.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/JetHT_label_separate_short_range.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/JetHT_subsystem_label.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/LastSubmission/JetHT_subsystem_label_short_range.png" width="400px" >
</p>

<p align="center">
    <img src="new_data/logs/minmaxscalar/JetHT_pc1.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/JetHT_pc2.png" width="400px" >
</p>

It's obviously to tell that the cluster of outlier are mainly consists of malfunction from MUON and TRACKER sub-detector. Not only the outlier that has an interesting patterns but clustering in inlier is also remarkably considerable as clustering mainly from malfunction of ECAL and HCAL that located near or inside the green band.

Please note that calculation of the matrix transform exclude failure scenario since it's a fake data and it might leading to a weird correlation in covariance matrix.

### Performance

#### 1) Include low statistics (Fill null with zero) and testing with only bad LS form human

Training with [feature set 1](../../data/new_prompt_reco/features.py)


* Autoencoder
  
<p align="center">
    <img src="new_data/logs/minmaxscalar/2e15BS12000EP/performance_EGamma_VanillaSparseContractiveVariational.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e15BS12000EP/performance_SingleMuon_VanillaSparseContractiveVariational.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e15BS12000EP/performance_ZeroBias_VanillaSparseContractiveVariational.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e15BS12000EP/performance_JetHT_VanillaSparseContractiveVariational.png" width="400px" >
</p>
The performance of AE for EGamma primary dataset is totally inefficient and even worse than randomly picking up which means that model even saw most of bad LS even looks better than many of good LS in the testing datasets. The rest of them is fairly acceptable but still not eought to exploit in the real system. Another interesting spot is the performance between couple of AE in SingleMuon PD.

* Extended Autoencoder
<p align="center">
    <img src="new_data/logs/minmaxscalar/2e15BS12000EP/performance_EGamma_SparseContractiveSparseVariationalContractiveVariationalStandard.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e15BS12000EP/performance_SingleMuon_SparseContractiveSparseVariationalContractiveVariationalStandard.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e15BS12000EP/performance_ZeroBias_SparseContractiveSparseVariationalContractiveVariationalStandard.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e15BS12000EP/performance_JetHT_SparseContractiveSparseVariationalContractiveVariationalStandard.png" width="400px" >
</p>

Even extended model has been combined various constrains that we known but it is still not improve any further in term of performance. Nevertheless, it has a remarkable stability especially for ContractiveVariational AE.

#### 2) Exclude low statistics and include bad LS from Failure Scenario for testing (Filter LS that has low EventsPerLs with value in the [settings](../../data/new_prompt_reco/setting.py))

Training with [feature set 2](../../data/new_prompt_reco/features_2.py)

* Autoencoder
<p align="center">
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/performance_SingleMuon_VanillaSparseContractiveVariational.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/performance_ZeroBias_VanillaSparseContractiveVariational.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/performance_JetHT_VanillaSparseContractiveVariational.png" width="400px" >
</p>

* Extended Autoencoder

<p align="center">
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/performance_SingleMuon_SparseContractiveSparseVariationalContractiveVariationalStandard.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/performance_ZeroBias_SparseContractiveSparseVariationalContractiveVariationalStandard.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/performance_JetHT_SparseContractiveSparseVariationalContractiveVariationalStandard.png" width="400px" >
</p>

We also perform an extended autoencoder for testing with this case, the above figure has shown the stability and smoothness of the threshold as we have seen in the previous feature selection.



* Reconstruction Error
  * EGamma

  * SingleMuon
    <p align="center">
        <img src="new_data/logs/minmaxscalar/2e16BS1200EP/avg_sd_Vanilla_SingleMuon_f2_1.png" width="400px" >
        <img src="new_data/logs/minmaxscalar/2e16BS1200EP/sum_sd_Vanilla_SingleMuon_f2_1.png" width="400px" >
    </p>
    The peak around feature 50 in good LS is qglobTkN. Secondly, the couple clump around feature 80 is qglobTkChi2. The next pile is qglobTkNHits as well as last forky shape in around feature hundred dominated by qMuNCh. 

  * ZeroBias
    <p align="center">
        <img src="new_data/logs/minmaxscalar/2e16BS1200EP/avg_sd_Variational_ZeroBias_f2_1.png" width="400px" >
        <img src="new_data/logs/minmaxscalar/2e16BS1200EP/sum_sd_Variational_ZeroBias_f2_1.png" width="400px" >
    </p>
    The residue in feature number 20 to 30 is qpVtxY. There are two huddles in bad LS where it mainly consists of qgTkPt, qgTkEta, and qgTkPhi. The clump in good LS around 70 to 80 mostly is qgTkPhi and qgTkN.

  * JetHT
    <p align="center">
        <img src="new_data/logs/minmaxscalar/2e16BS1200EP/avg_sd_Variational_JetHT_f2_1.png" width="400px" >
        <img src="new_data/logs/minmaxscalar/2e16BS1200EP/sum_sd_Variational_JetHT_f2_1.png" width="400px" >
    </p>
    The features that contain a very first peak in bad LS is qpVtxChi2. Secondly, around feature number 80 to 90 are qPFMetPt and qPFMetPhi. Lastly, there are the last two chunks of features (~120-127 and ~130-145) that behave like a noisy for both good and bad LS. Highly correlated features that show similar features (~15-35) are qpVtxX, qpVtxY, and qpVtxZ.

### Distribution of decision value (to find the threshold)
<p align="center">
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/se_dist_Vanilla1f2_SingleMuon_unlog.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/se_dist_Variational1f2_ZeroBias_unlog.png" width="400px" >
    <img src="new_data/logs/minmaxscalar/2e16BS1200EP/se_dist_Variational1f2_JetHT_unlog.png" width="400px" >
</p>


For the weekly report which contain the full detail of this study please checkout [this direcotry](new_data/reports/).