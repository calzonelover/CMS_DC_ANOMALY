# Semi-supervised Learning (PromptReco)
* Unsupervised Models
  * Schölkopf's One-Class SVM
  * Isolation Forest
  * 4 Flavours of Autoencoder
* Feed only good LS for train and validate the model
* Testing with good LS and bad LS
* Consequently, it’s falling into **Semi-supervised Learning** category

## Autoencoder (AE)
* Truncated normal initializer
    
    For model weight initializer, we are using truncaed normal initializer which basically you take a gaussian distribution and putting the cutoff only inside $\pm2\sigma$ to prevent some high absolute value that might leading to divergence of model in the training process.

    In our case, we set up $\sigma=1$ and $\mu = 1$
<p align="center">
    <img src="../static/img/normal_dist.png" width="500px" >
    <br>
    <em>Gaussian distribution, retrieved from https://towardsdatascience.com/understanding-the-68-95-99-7-rule-for-a-normal-distribution-b7b7cbf760c2</em>
</p>

* Adam optimizer

    Adam stands for **adaptive moment estimation**. Basically, it's combine Momentum optimization and RMSProp to keep the residue from gradients decaying from the previous one.

    With configuration: $lr = 0.2$ (learning rate), $\beta_1 = 0.7$ and $\beta_2 = 0.9$

    Ref: Adam: A Method for Stochastic Optimization, D. Kingma, J. Ba (2015)

### Vanilla AE
<p align="center">
    <img src="../static/img/vanilla_ae.png" width="300px" >
    <br>
    <em>Body of Vanilla AE</em>
</p>

* Concise the information into small latent space and reconstruct
* Loss function is defined by
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{tot}}&space;\equiv&space;\frac{1}{N}\sum_i^N&space;|x_i-\tilde{x}_i|^2" title="\mathcal{L}_{\text{tot}} \equiv \frac{1}{N}\sum_i^N |x_i-\tilde{x}_i|^2" />
</p>

### Sparse AE
* Tweak by L1 Regularizaion (Prevent overfitting)
* Loss function
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{tot}}&space;\equiv&space;\frac{1}{N}\sum_i^N&space;|x_i-\tilde{x}_i|^2&space;&plus;&space;\lambda_{\text{s}}\sum_j||w_j||" title="\mathcal{L}_{\text{tot}} \equiv \frac{1}{N}\sum_i^N |x_i-\tilde{x}_i|^2 + \lambda_{\text{s}}\sum_j||w_j||" />
</p>

* where 
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\lambda_{\text{s}}&space;=&space;10^{-5}" title="\lambda_{\text{s}} = 10^{-5}" />
</p>

### Contractive AE
* Tweak by Jacobi Matrix (Prevent variation in dataset)
* Loss function
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{tot}}&space;\equiv&space;\frac{1}{N}\sum_i^N&space;|x_i-\tilde{x}_i|^2&space;&plus;&space;\lambda_{\text{c}}||J_h(x)||^2" title="\mathcal{L}_{\text{tot}} \equiv \frac{1}{N}\sum_i^N |x_i-\tilde{x}_i|^2 + \lambda_{\text{c}}||J_h(x)||^2" />
</p>

* where 
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\lambda_{\text{c}}&space;=&space;10^{-5}" title="\lambda_{\text{c}} = 10^{-5}" />
</p>

* Definition of Jacobi matrix 
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?||J_h(x)||^2&space;\equiv&space;\frac{1}{N}\sum_{ij}\left(\frac{\partial&space;h_j}{\partial&space;x_i}\right)^2" title="||J_h(x)||^2 \equiv \frac{1}{N}\sum_{ij}\left(\frac{\partial h_j}{\partial x_i}\right)^2" />
</p>

* Activation function in our case
  
  * PReLu activation function
    <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?||J_h(x)||^2&space;=&space;\frac{1}{N}\sum_i^N\sum_j[\alpha_j&space;H(-(w_{jk}x^{ik}&plus;b_j))&space;&plus;&space;H(w_{jk}x^{ik}&plus;b_j)]\sum_k(w_{jk})^2" title="||J_h(x)||^2 = \frac{1}{N}\sum_i^N\sum_j[\alpha_j H(-(w_{jk}x^{ik}+b_j)) + H(w_{jk}x^{ik}+b_j)]\sum_k(w_{jk})^2" />
    </p>
  * Sigmoid activation function
    <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?||J_h(x)||^2&space;=&space;\frac{1}{N}\sum_{ij}[h_{ij}(1-h_{ij})]\sum_k(w_{jk})^2" title="||J_h(x)||^2 = \frac{1}{N}\sum_{ij}[h_{ij}(1-h_{ij})]\sum_k(w_{jk})^2" />
    </p>
    

### Variational AE
<p align="center">
    <img src="../static/img/variational_ae.png" width="300px" >
    <br>
    <em>Body of Variational AE , Image revised from https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf </em>
</p>

* Random “new sampling” in latent space by gaussian random generator
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{Z}&space;\equiv&space;\mathcal{N}(\mu_i,&space;\sigma_i)" title="\mathcal{Z} \equiv \mathcal{N}(\mu_i, \sigma_i)" />
</p>

* Tweak by reduce discontinuity in latent space
* Loss function
  
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{tot}}&space;=&space;\frac{1}{N}\sum_i^N&space;|x_i-\tilde{x}_i|^2&space;&plus;&space;\mathcal{D}_{\text{KL}}(p|q)" title="\mathcal{L}_{\text{tot}} = \frac{1}{N}\sum_i^N |x_i-\tilde{x}_i|^2 + \mathcal{D}_{\text{KL}}(p|q)" />
</p>

* Since we represent latent space by applying gaussian. Kullback-Leibler Divergence term would looks like
  <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathcal{D}_{\text{KL},&space;i}&space;=&space;\frac{1}{2}\sum_k^{n_{\text{latent}}}(\mu_{ik}^2&space;&plus;\sigma_{ik}^2&space;-&space;2\log\sigma_{ik}&space;-&space;1)" title="\mathcal{D}_{\text{KL}, i} = \frac{1}{2}\sum_k^{n_{\text{latent}}}(\mu_{ik}^2 +\sigma_{ik}^2 - 2\log\sigma_{ik} - 1)" />
  </p>
  In order to minimize the KL-div's term, the model have to adapt it's origin of random sampling toward nearly the origin of the latent space and have to adapt the sigma to one to minimize this term. In principle, outlier would located quite far from the origin and might eventually not sitting around the dense cluster of inlier.

* Then total loss function would looks like
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{tot}}&space;=&space;\frac{1}{N}\sum_i^N&space;|x_i-\tilde{x}_i|^2&space;&plus;&space;\frac{1}{2N}\sum_i^N\sum_k^{n_{\text{latent}}}(\mu_{ik}^2&space;&plus;\sigma_{ik}^2&space;-&space;2\log\sigma_{ik}&space;-&space;1)" title="\mathcal{L}_{\text{tot}} = \frac{1}{N}\sum_i^N |x_i-\tilde{x}_i|^2 + \frac{1}{2N}\sum_i^N\sum_k^{n_{\text{latent}}}(\mu_{ik}^2 +\sigma_{ik}^2 - 2\log\sigma_{ik} - 1)" />
</p>

### Mother Class for various AE inheritance
Since we are using Tensorflow(v1.13), the concept of abstract graph connection and session execution require a comprehension to maximize the profit of parallelizable task.

Generally, we could have a single graph and session for the model as a global session and global graph connection with all variables you have. In case you have multiple model, it's likely that we want an isolate graph and session for various model execution to make sure that it doesn't share the same variable. That is why we need OOP concept to take care all of those.

The figure below is [the mother class](NN/base.py) of our AE since all the model need their own graph and session as well as other utility function that helps our life easier to have less code and more scalable. 
<p align="center">
    <img src="../static/img/baseclass_nn.png" width="200px" >
    <br>
    <em>Main components of the AE's base class</em>
</p>

Not only the main utility function that we could inherit from mother class to have various child class but we could also combine multiple contrains as we want which has been done in the extended model that located in [this script](reco/new_autoencoder.py).