# 2018 Dataset
\-

# 2016 Dataset (Only JetHT)

### Primary Analysis
In order to roughly understand the clustering (similar patterns) of data, one way to do it is to reduce the dimension of data. In our case, there are 259 features which be transformed into 2 dimension on the basis of 2 eigenvectors that belonging to covariance matrix that computed from the datasets
<p align="center">
    <img src="Old_Data/logs/ReducedFeatures/minmaxscalar/JetHT_label.png" width="500px" >
    <br>
    <em>Principal component with the labeled color from the system</em>
</p>

As you could see on the green line that there are nice tubular shape which is good LS and a few weird LSs that located outside the tubular shape as well as bad LS that could be divided into the bad LS with some patterns and anamaly bad LS which I would called both of them as "outlier". That's essentially the punchline why I called outlier detection instead of anomaly detection.

### Performance
We Iteratively retrain the model 10times to make sure that it's working systematically and plot the root mean square as the shady fluctuation in the following figures
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
    <br>
    <em>Various AE</em>
</p>
For Vanilla AE, the contamination of bad LS falling into good LS is around 1% over the good LS below the cutoff and there are only countable of good LS falling into bad LS which might be ignorable.

For OneClass-SVM, the contamination LS that bad falling into good LS is almost exactly the same as Vanilla AE does. There is no coincidence for totally different approach of model train and spot the same thing. This might implicitly implies that it either came from some imperfection of data in the training and testing or some kind of malfunction in the sub-system couldn't propagated into JetHT physics objects.

As can be seen in the distribution, there is no clear grey zone for this study so far.

### Example of square error from reconstruction
Here are the example of LS reconstruction which calculated from x and x' between good and bad LS
<p align="center">
    <img src="Old_Data/logs/ReducedFeatures/minmaxscalar/BS256_EP1200_noshuffle/example_se.png" width="500px" >
    <br>
    <em>Reconstruction error from Vanilla AE</em>
</p>

### Extended Investigation
You might wondering why many of bad LS seems to have a group of bad LS as you have seen in the plot of hyperspace and few collection of bad LS in decision value distribution. In this section, I want to explicitly prove that the model really see that the right cluster is the worse bad LS and more closer to tubular is less bad LS which decision value have to be quite similar to good LS. In order to prove that, I choose our best candidate to shading the decision value as z-axis color to represent how bad LS in each data point is.
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