# Non-parametric Bayesian Prior for Variational Auto-encoder

This is the final project for course *Advanced Topics in Bayesian Statistics*.
You can find the mid-term presentation about hierarchical dirichlet process [here](http://myemacs.com/2017/04/06/Hierarchical-Dirichlet-Process/).

 The whole idea for this work comes from this paper by Prof. Eric Xing's team:

 > [Nonparametric Variational Auto-encoders for Hierarchical Representation Learning](https://arxiv.org/abs/1703.07027)

 This is the abstract for this paper:

 > The recently developed variational autoencoders (VAEs) have proved to be an effective confluence of the rich representational power of neural networks with Bayesian methods. However, most work on VAEs use a rather simple prior over the latent variables such as standard normal distribution, thereby restricting its applications to relatively simple phenomena. In this work, we propose hierarchical nonparametric variational autoencoders, which combines tree-structured Bayesian nonparametric priors with VAEs, to enable infinite flexibility of the latent representation space. Both the neural parameters and Bayesian priors are learned jointly using tailored variational inference. The resulting model induces a hierarchical structure of latent semantic concepts underlying the data corpus, and infers accurate representations of data instances. We apply our model in video representation learning. Our method is able to discover highly interpretable activity hierarchies, and obtain improved clustering accuracy and generalization capacity based on the learned rich representations.

 I also give a presentation about this paper and you can find slides [here](http://myemacs.com).

 This project also proposes a combination of non-parametric bayesian prior and variational auto-encoder.
 In the normal variational auto-encoder, we assume that the latent variable has the non-informative prior.
 And the object consists two parts: the log-likelihood under the variational distribution and the similarities between posterior and the prior (KL divergence term).
 For more details about variational auto-encoder you can read [this tutorial](kvfrans.com/variational-autoencoders-explained/).

 But in some cases it is not suitable to use the same prior regardless of the real data we used.
 Although with the powerful encoder and decoder it may be fine, it is still non-informative.
 One idea of the paper listed above is to use the prior for the hierarchical clustering.
 The model they used is nCRP which is a infinitely large tree.
 In each node of the tree they did a Chinese Restaurant Process.

 Instead of using a complex model to group the data into a big tree, i tried to use some simple non-parametric bayesian model as prior.
 The model I used here is the dirichlet process mixture model.
 You can think it as a mixture models with infinite number of components (it is totally depends on hyperparameters and the training data).
 Compared to the classical Gaussian Mixture Models we don't have a pre-defined value for the number of components.
 For more details about dirichlet process and non-parametric bayesian stuff I found [this video seris](https://www.youtube.com/watch?v=kKZkNUvsJ4M) given by Prof. Tamara Broderick is very helpful for the beginner because of the vivid examples.

 ## Experiments
 ------

 ### Corpus

 What is used for experiments is [MNIST of handwritten digits](http://yann.lecun.com/exdb/mnist/).
 Following is some description about this corpus:

 >The MNIST database of handwritten digits, available from this page, has a training set of ***60,000*** examples, and a test set of ***10,000*** examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
 >The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the ***28x28*** field.

 The corpus size is not too small for practice in machine learning and is not too large for non-parametric models.

 Since we have ground truth (10 classes: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), I did two class of experiments:

  * As the indefinite mixture model, how well is the clustering result? For example, how pure is each cluster? Does one cluster includes samples from multiple class?
  * As the auto-encoder model, how well is the reconstruction result? Further more, can we sample from this model?

 ### Experiment Details
 Variational inference is used for dirichlet process.
 Compared to sampling method like Gibbs sampling variational inference is usually faster.
 For more details about variational inference implementation for dirichlet process please read [this page](http://scikit-learn.org/stable/modules/dp-derivation.html).
 I chose diagonal covariance in all experiments.

 The whole training process repeats two steps:
 * Fixed the encoder, sampled <img src="https://latex.codecogs.com/gif.latex?$z$" title="$z$" /> from the training data and update the prior
 * Fixed the prior and update the encoder and decoder as the standard variational auto-encoder. The KL-divergence between the posterior and the predicted cluster (which is still a normal distribution) is calculated as well as the likelihood.

 And the whole system is trained with 11 iterations.

 For the encoder, one hidden layer with 400 nodes is used to transform the image data. And then two layers are added to map the output to mean and variance.

 For the decoder, one hidden layer with 400 nodes is also used, and then one hidden layer with 28*28 nodes is added for reconstruction.

 All results are given on the test set.

 ### Clustering results analysis

 In this part I am trying to figure out the "purity" of the clustering.
 For example, we think this clustering result is "pure":

  |Class labels | 1 | 1 | 1 | 2 | 2 | 3 | 3 | 3 | 3 |
  |-------------|---|---|---|---|---|---|---|---|---|
  |Cluster labels | 1 | 1 | 2 | 3 | 3 | 4 | 4 | 5 | 5 |

 How to estimate the clustering result?
 Fortunately as the "purity" suggests we use the [mutual information](https://en.wikipedia.org/wiki/Mutual_information).

 For the example above, the mutual information is about 1.06.
 Because the mutual information is between 0 (independent) and positive infinity, it is not clear whether is good or not.
 Some good alternatives include [normalized mutual information](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score) which is between 0 and 1.
 For this example the normalized mutual information is approximately 0.82, fairly good.

 #### Baseline Performance

 For the baseline system, dirichlet process is directly used. The maximum number of components is set to 50 and generally it is very slow and hard to converge.

 The mutual information is around 1.13 when the maximum number of iterations is set to 1000 and number of initializations is set to 5.

 #### Relation between latent space dimension and mutual information

 In this experiment the maximum number of dimension is fixed to 100.

 The hidden dimension is varied from 5 to 50.

 ![Figure 1](https://github.com/bobchennan/VAE_NBP/raw/master/vis2.png "Connection between latent space dimension and mutual information")

 It is quite clear that 15 is the best configuration which means either too small or too large latent space is not good.

 #### Relation between maximum number of components and mutual information

 One drawback of variational inference implementation for dirichlet process is that the maximal number of components need to be assigned manually for truncation.
 In the [original paper](http://www.cs.columbia.edu/~blei/papers/BleiJordan2004.pdf) they truncated the variational distribution instead of using the whole stick-breaking representation.

 So one interesting question is that how the maximum number of components influence the result.
 I tried different values from 50 to 700 and the latent dimension is fixed to 15.

 This is the result:

![Figure 2](https://github.com/bobchennan/VAE_NBP/raw/master/vis1.png "Connection between maximum number of components and mutual information")

As you can see, when this number is larger than 500, there is not much improvement.

 ### Sample analysis
 In order to sample from the model, the latent vector is sampled from the prior (from either non-informative prior or dirichlet process mixture model).
 Then the decoder will transform the latent vector to the real image.

 Samples from variational auto-encoder with dirichlet process mixture model prior:

![Figure 3](https://github.com/bobchennan/VAE_NBP/raw/master/gen_dp.png "Samples from variational auto-encoder with dirichlet process mixture model prior")

Samples from variational auto-encoder with non-informative prior:

![Figure 4](https://github.com/bobchennan/VAE_NBP/raw/master/gen_normal.png "Samples from variational auto-encoder with non-informative prior")

As you can see, samples from the NBP prior is much better than non-informative prior as expect. It is well known that variational auto-encoder sampling suffers blurry problem. Dirichlet process mixture model cannot fully solve this problem but the result is much better.

 ## Future Work
 ------
 While I am working on this project I realized dirichlet process may be not a good choice in the mutual information perspective. It is well known that in the dirichlet process, richer mixture get richer. So it tends to have some large mixtures which contains samples from all classes. This is not good for both mutual information and sampling in reality.

 For alternatives, maybe the repulsive models are a good choice. Depends on the base measure, mixtures in my experiments are too closed to each other. That is the reason that repulsive models may be a good choice.

 ## code
 The implementation can be found in my [github repo](https://github.com/bobchennan/VAE_NBP).
