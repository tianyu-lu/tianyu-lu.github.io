---
layout: post
title:  "Optimizing Plastic Degrading Proteins"
date:   2019-11-01 10:00:00
categories: iGEM Protein ML
excerpt_separator: <!--more-->
---

We present a generalizable and automated pipeline for protein design. Our model can be applied to the optimization of any protein class, even those with scarce data.

<!--more-->
## 1. Overview

Our model consists of an AdaBoost regressor that is able to predict a protein property from sequence alone. We then train a recurrent neural network (RNN) that is able to generate novel protein sequences. Generated sequences are evaluated by the regressor and those that pass a specified threshold are added in the training set for the RNN to be retrained. This iterative process continues until convergence or experimental validation.

## 2. Generative Model
### Intuition
We will use a Recurrent Neural Network as a generative model. The structure of an RNN is shown below. The $$\mathbf{x}_i$$ are the input vectors, the boxes in $$A$$ are described in more detail in the Formal Definition section, and the $$\mathbf{h}_i$$ are the outputs for each cell.

<center>
    <img src="https://i.imgur.com/zMIkiPH.png" width="250px" />
</center>

These types of neural networks are well-suited for sequence data, such as amino acid sequences. We were first drawn to the generative ability of RNNs from an experiment done by Andreji Karpathy. He trained an RNN on the entire Shakespeare corpus and asked it to generate new Shakespeare text. Remarkably, the sample shown below closely captures Shakespeare's writing style.  

<center>
    <img src="https://i.imgur.com/PDt73bz.png" width="250px" />
</center>

He also trained an RNN on Linux source code.  

<center>
    <img src="https://i.imgur.com/vJ7KXxI.png" width="250px" />
</center>

Disregard the particular words but notice the structure of the generated code shown below. The RNN was able to capture the syntax of the language remarkably well. Notice how in this example, every open parentheses is eventually followed by a close parentheses. These two experiments inspired us to apply RNNs to generate novel protein sequences. In effect, we are asking the question, how well can RNNs capture patterns in the protein language, the ordered sequence of amino acids?  
In the software industry, RNNs are frequently used in chatbots, where an objective might be to predict the next word to generate. Suppose we want to predict the red word. Its tense actually depends on a noun many words before it. This concept is known as long term dependencies. 

<center>
    <img src="https://i.imgur.com/9QT1wVS.png" width="250px" />
</center>

Thus, an RNN must be able to remember data seen previously, often for long distances, and a variant of RNN called LSTM, does this particularly well. What does this have to do with proteins? Proteins are structures in 3D space. Thus, it is rich in regions that are close together in 3D space but its residues are far apart in primary sequence.

<center>
    <img src="https://i.imgur.com/ljsHcgr.png" width="230" height="200" />
</center>

The intuition is that the ability of LSTMs to capture these long term dependencies will work well for protein sequence generation.


### Formal Definition
An LSTM is defined as follows:  

<center>
    <img src="https://i.imgur.com/6FIf8M3.png" width="250px" />
</center>

All variables are vectors except for the weight matrices $$W$$. The notation of $$[a, b]$$ means to stack the column vectors of $$a$$ and $$b$$ vertically. $$\sigma()$$ is the sigmoid activation function.  

The result of (1), $$\tilde{c}^{<t>}$$, can be interpreted as the data to remember for the current cell. Equations (2), (3), and (4) output vectors which can be thought of as gates, where (2) is the update gate, (3) is the forget gate, and (4) is the output gate. Since $$\sigma(z) \in (0, 1) \forall z \in \mathbb{R}$$, the gamma scalars can be thought of as percentages. In (5), we multiply $$\tilde{c}^{<t>}$$ by $$\Gamma_u$$ element-wise to determine what percentage of each element of the new data to remember. We also multiply $$c^{<t-1>}$$, the memory state of the previous time step, by $$\Gamma_f$$ element-wise to determine what percentage of the previous memory to forget. Then, we see that $$c^{<t>}$$ is a combination of old and new memory, where their proportions are determined by learnable gates. Finally, we need to determine the input to the next cell by equation (6).  

Thus, we can see that at each time step, the input data is $$x^{<t>}$$, the main output of each cell is $$a^{<t>}$$, and the $$c^{<t>}$$ is the memory state that is able to capture long term dependencies.

### Code
The code for sequence generation was adapted from [here.](https://www.tensorflow.org/tutorials/text/text_generation) For training data, we collected 19 PETase sequence function pairs through a literature search. Parsed sequences are available [here](https://github.com/MauriceR71/UniRep) as `petase_seqs.txt` We wrote a wrapper function, `run_RNN()` that generated new protein sequences after training on a file specified in its input parameters. Our function only returns sequences that passes a set of filters, defined in the function `passes_filters()`. These filters include  
1. Enforcing length of sequence to be no more than 20 amino acids away from the length of wildtype PETase.
2. Enforcing the instability index, as determined by the [DIWV matrix](http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=2075190&dopt=Abstract), to be less that 40.0
3. Enforcing the predicted activity of the sequence to be at least 2.5 fold higher than wildtype activity.

A list of sequences that pass all filters are returned by `run_RNN()`. The point of these filters is to compensate for the uncertainties of the discriminative model.

## 3. Discriminative Model
### Intuition
A sequence to function model allows us to predict the fitness of a protein sequence without time-consuming lab experiments. A key property that such a model must have is that it's generalizable to unseen regions of sequence space. This is a central problem in machine learning, where we would like a model that fits well to data it has seen (training data), and at the same time also fits well to data it has not seen (validation and test data).  
In particular, we don't want the model to overfit, i.e. an overcomplicated model. An analogy can be a student preparing for a math test by memorizing the answers to homework questions. Clearly such a student would not perform well on the test. Also, we don't want the model to underfit, i.e. a model that is too simple.

### Model Evaluation
As explained in the intuition, our goal is to search for a model that has low validation and test errors. A nice way to present this is by plotting the actual catalytic activity versus the predicted activity. A perfect model would coincide the line $$y=x$$. We can measure this using the correlation coefficient, $$R^2$$.

### UniRep
Unlike the RNN where the training set can be any text file, the inputs to an oracle must be processed to be of the same size. Unfortunately, such a task for protein sequences is nontrivial. As an example, the hydrolase class of protein sequences have lengths that range from 200 to 1400 amino acids. For a pipeline to be generalizable, it must be able to convert all protein sequences to the same size. Such a problem can be again solved by RNNs. Notice that the size of the output vector for each unit is the same. The size of the protein sequence then, is controlled by the number of recurrent cells, one for each amino acid. More interestingly, Alley et al. found in [their paper](https://www.biorxiv.org/content/biorxiv/early/2019/03/26/589333.full.pdf) that some neurons in these protein encoding RNNs actually encode information about the sequence's secondary structure, while their training set included no information about secondary structure. Their model is called UniRep and turns any protein sequence of any length into a 64-dimensional vector. 

### Final Model - AdaBoost
After comparing the performances of many model types (LSTM, SVM, GP, CNN, Linear), we found that AdaBoost regressors demonstrated excellent performance on the training set and decent performance on the validation set. We trained a series of AdaBoost regressors on 60% of the 19 PETase sequences and tested their performance on the remaining 40%. We explore the model performance on varying number of estimators.
<center>
    <img src="https://i.imgur.com/Rsm9skH.png" width="250px" />
</center>

The AdaBoost model has the advantage that it fits the training data rapidly. The error on the test data is low, considering that those were sequences never seen before, and that the range of possible predictions are from 0 to 3.4. Taking a look at the 19th regressor, we see the following:

<center>
    <img src="https://i.imgur.com/fk0RN7P.png" width="250px" />
</center>

The $$R^2$$ for the training data is 0.9986 and the $$R^2$$ for the testing data is 0.5378.   

As a further test, we computed $$k$$-fold cross-validation loss for $$k=1$$. 

<center>
    <img src="https://i.imgur.com/26oMwGp.png" width="250px" />
</center>

Here, the cross validation loss serves as a proxy for the average uncertainty in the model's predictions. It is important to keep the uncertainty of the model in mind while evaluating sequence function. 

### AdaBoost Intuition
We chose to use an AdaBoost regressor for three reasons. The first and most important reason is that it achieved near zero training loss and decent validation loss. Two, AdaBoost, as suggested in its name, is adaptive. That means as new data comes in, it is able to quickly adjust its predictions. Third, AdaBoost is highly scalable. Many real time face detection software rely on AdaBoost's fast prediction outputs for large data inputs. Here's the intuition on how AdaBoost achieves all this.  

For simplicity, let's assume we wish to train a model that outputs if an image shows a cat or a dog, i.e. a binary classification problem. A weak classifier is a model that performs slightly better than chance. So in this case, a random classifier would be expected to have a misclassification rate of 50%. A weak classifier would achieve a misclassification rate of slightly lower than 50%. A common and remarkable theme in statistics is that a crowd is in general more accurate than an individual. AdaBoost's goal is to let a bunch of weak classifiers vote, and the output is the result of this group decision making.  

The algorithm starts with one weak classifier, evaluates its predictions, and collects the examples that were misclassified. It then trains a second weak classifier, but this time, penalizing those misclassified examples more. Intuitively, the algorithm is learning from its mistakes.

### AdaBoost Formal Definition
Let the input data be $$\mathcal{D}_N = \{ \mathbf{x}^{(n)}, t^{(n)} \}_{n=1}^N$$, where $$t^{(n)} \in (-\inf, +\inf)$$. A classifier $$h$$ maps input vectors to their values, i.e. $$h:\mathbf{x}\rightarrow (-\inf, +\inf)$$. Define a loss function, in our case we used the linear loss function, as follows:  

<center>
    <img src="https://i.imgur.com/iKPNRT4.png" width="230" height="70" />
</center>  

where the denominator ensures the loss is in $$[0, 1]$$. The following is the AdaBoost algorithm. The formulation is slightly different from that in the original paper, but the idea remains the same.  

Let each sample have its associated weight, $$w^{(n)} = \frac{1}{N}$$, where $$n = 1, \cdots , N$$. Let $$T$$ be the number of estimators. Then for $$t = 1, \cdots , T$$, do the following:  

1. Let $$h_t = \text{argmin}_{h}\sum\limits_{n=1}^{N}w^{(n)}\mathcal{L}(h)$$
2. Let $$\text{err}_t = \Bigg[\sum\limits_{n=1}^{N}w^{(n)}\mathcal{L}(h_t)\Bigg] \Big/ \Bigg[\sum\limits_{n=1}^{N}w^{(n)}\Bigg]$$, the weighted error.
3. Let $$\alpha_t = \frac{1}{2}\text{log}\frac{1 - \text{err}_t}{\text{err}_t}$$, the classifier coefficient. $$(\alpha \in (0, \infty))$$
4. Update weights by $$w^{(n)} \leftarrow w^{(n)}\text{exp}\Big( 2\alpha_t \mathbb{I}\{ h_t(\mathbf{x}^{(n)}) \neq t^{(n)} \} \Big)$$

The final regressor is a linear combination of the weak learners by $$H(\mathbf{x}) = \sum\limits_{t=1}^{T}\Big( \alpha_t h_t (\mathbf{x}) \Big)$$

$$h_t$$ is the weak classifier. Its examples are weighted by the set of weights $$w^{(n)}$$, where a higher $$w^{(n)}$$ increases the loss $$\mathcal{L}(h)$$ more, making those examples weighted more heavily.   
$$err_t$$ is the error rate for that particular weak classifier, $$h_t$$ found in step 1. The denominator is there to normalize $$err_t$$ between 0 and 1.  
$$\alpha_t$$ is a measure of how accurate the particular classifier, $$h_t$$ is. If $$err_t$$ is low, i.e. near 0, $$\alpha_t \rightarrow +\infty$$, indicating a very accurate model. If $$err_t$$ is high, i.e. near 0.5, $$\alpha_t \rightarrow 0$$, indicating an inaccurate model.  
In step 4, the indicator function $$\mathbb{I}\{ h_t(\mathbf{x}^{(n)}) \neq t^{(n)} \}$$ is 1 if the prediction is wrong, and is 0 if the prediction is correct. Thus, weights increase when the labels are misclassified.


## 4. Optimization Algorithm
Equipped with a generative model suitable for protein sequence generation, and a decent discriminative model for protein sequence evaluation, we can build our optimization workflow, shown below.
<center>
    <img src="https://i.imgur.com/e9OSZ7i.png" width="250px" />
</center>

First generate sequences based on existing sequence to function data. Then, score those sequences and only keep those that score beyond a preset threshold. Finally, add the high scoring sequences back into the training set and retrain the RNN to generate new sequences. Iterate until convergence or experimental verification.

### Code
[Pipeline code](https://colab.research.google.com/drive/15bxbkmiGjE7pxR3TXLsIZo9sGitOekSG)  

[Discriminator code](https://colab.research.google.com/drive/1ZK7SePe4I_bwJNI2CHq8EAaZm3Fw6yiO)  

Comments are provided in the notebooks on how to reproduce results in this article.


## 5. Future Directions
### Transfer Learning
Biswas et al. developed a composite residues model for the prediction of GFP function from sequence. We trained an [implementation](https://github.com/krdav/Composite-Residues_keras_implementation) of their model on 80% of about 58000 GFP sequences. We then retrained the last two layers of the neural network on PETase sequences. Due to the much smaller dataset for PETase, we observe chaotic behaviour in the validation loss.

<center>
    <img src="https://i.imgur.com/iiJPLpE.png" width="250px" />
</center>

After transfer learning, the validation loss becomes much more controlled:

<center>
    <img src="https://i.imgur.com/Sax60x6.png" width="250px" />
</center>

As a test of the applicability of transfer learning, we incorporated it as part of an oracle in the design algorithm described by Brookes et al [here](https://arxiv.org/pdf/1901.10060).   

Though GFP is far from a hydrolase, its strength is in the amount of data available. A classic example in computer vision is that you can first train a neural network on millions of cat pictures on the internet, then fine tune on much smaller datasets of radiology images.

<center>
    <img src="https://i.imgur.com/Z1tFhyi.jpg" width="250px" />
</center>

The network can identify the low-level features like edges and texture in the first few layers, which we can reuse for training models for cancer detection. Perhaps we can first find patterns underlying all protein sequences, then fine tune to specific classes of proteins like PETase. In effect, transfer learning allows the oracle to make predictions based on much more data than just 19 sequences since it was first trained on 58000 GFP sequences.  

### Conditioning by Adaptive Sampling

The idea behind their algorithm is to condition the probability distribution of protein sequences such that sampling from this conditioned distribution is more likely to yield a stable protein that has a property of interest.   

Instead of an RNN, a variational autoencoder (VAE) is used as a generative model, in particular because by its definition, generating from a VAE involves sampling from a multivariate probability distribution, in this case, ones that encode for protein sequences. For a detailed treatment of VAEs, Serena Leung gives a nice derivation [here.](https://youtu.be/5WoItGTWV54?t=1176)  

Another important feature of their model is that they use a Gaussian Process in tandem with an oracle, to avoid falling into "pathological regions" of input space. Due to the incredibly sparse data of available PETase mutant sequences, it is unwise to rely completely on a model based on those to make decisions. A Gaussian Process is able not only make predictions of protein function from sequence, but also quantifies the uncertainty of those predictions. The intuition behind it is that we should trust our data. So where there is data, there is less uncertainty, and where there is less data, there is more uncertainty. For a more formal definition, Nando de Frietas' lectures [here](https://www.youtube.com/watch?v=4vGiHC35j9s&t=1s) is an excellent resource.   

A sequence alignment of this algorithm's final output to the wildtype PETase is show below:  

<center>
    <img src="https://i.imgur.com/2jcHjGP.png" width="250px" />
</center>

For the mutated residues, notice the large number of ':' which denote residues with strongly similar properties. Each colour denotes a different residue type, and the alignment somewhat shows local conservation among residue types. Globally, it appears the distribution of residue frequencies is also conserved, with red (hydrophobic) and green (hydroxy/sulfhydryl/amine) taking the majority and blue (basic) taking the minority.

### Code

[Conditioning by Adaptive Samping code](https://colab.research.google.com/drive/11JI0PEvFY9hVLa-U5kzkN9Ry-LPbm8O9) adapted from Brookes et al.

### LSTM Discriminator

We found an interesting train/test loss plot from a 16-unit LSTM regression model that may be worth exploring further:
<center>
    <img src="https://i.imgur.com/i6BZHoL.png" width="250px" />
</center>

where Hydrophobic embeddings are encoded by the indices, starting at 1, of the following ordered string `FIWLVMYCATHGSQRKNEPD`, which is all the amino acids ordered from most to least hydrophobic at pH 7. For example, `M` would be encoded as a 6.

### UniRep

We could further explore the usefulness of UniRep as inputs to protein function prediction models by comparing model performance with and without UniRep preprocessing. In addition, given the near zero test loss for the AdaBoost model, investigating the differences between UniRep inputs could increase model interpretability.

## 6. References

Alley, E. C., Khimulya, G., Biswas, S., AlQuraishi, M., & Church, G. M. (2019). Unified rational protein engineering with sequence-only deep representation learning. bioRxiv, 589333. [Link](https://www.biorxiv.org/content/biorxiv/early/2019/03/26/589333.full.pdf)

Bepler, T., & Berger, B. (2019). Learning protein sequence embeddings using information from structure. arXiv preprint arXiv:1902.08661. [Link](https://arxiv.org/pdf/1902.08661)

Biswas, S., Kuznetsov, G., Ogden, P. J., Conway, N. J., Adams, R. P., & Church, G. M. (2018). Toward machine-guided design of proteins. bioRxiv, 337154. [Link](https://www.biorxiv.org/content/biorxiv/early/2018/06/02/337154.full.pdf)

Brookes, D. H., Park, H., & Listgarten, J. (2019). Conditioning by adaptive sampling for robust design. arXiv preprint arXiv:1901.10060. [Link](https://arxiv.org/pdf/1901.10060)

Guruprasad, K., Reddy, B.V.B. and Pandit, M.W. (1990) Correlation between stability of a protein and its dipeptide composition: a novel approach for predicting in vivo stability of a protein from its primary sequence. Protein Eng. 4,155-161. [Link](http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=2075190&dopt=Abstract)

Karpathy, A. (2015, May 21). The Unreasonable Effectiveness of Recurrent Neural Networks. Retrieved October 15, 2019. [Link](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## 7. Supplementary Information

### Files

Repo 1: 
https://github.com/MauriceR71/UniRep

`AdaBoostRegressor_18_2.pkl`: The best performing discriminative model described above.  

`petase_seqs.txt`: The 19 PETase sequences that serves as the training set for the RNN  

`blast.txt`: 250px sequences most similar to wildtype PETase obtained by a BLAST search. This may also be used as a training set for the RNN  

`petase_seqvecs.npy`: Numpy array of shape (19, 64, 3) that is the output of UniRep's `get_rep()` function for 19 PETase sequences.  

`petase_vals.npy`: Numpy array of the corresponding activity values of the 19 PETase sequences

`data/diwv.csv`: The matrix used to calculate instability  

Repo 2:
https://github.com/MauriceR71/Composite-Residues_keras_implementation

`hydrolase_all_vecs.npy`: Numpy array containing UniRep vectors of 6182 hydrolase sequences obtained from the SABIO-RK kinetics database  

`hydrolase_all_vals.npy`: Numpy array of hydrolase sequence activity corresponding to the sequences above  

`gfp_data.csv`: .csv file of about 58000 GFP sequences and their corresponding brightness  

`PETase_mutations.csv`: .csv file of 19 PETase sequences and their corresponding activity  
