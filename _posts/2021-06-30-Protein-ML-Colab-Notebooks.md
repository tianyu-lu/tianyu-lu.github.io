---
layout: post
title:  "Protein ML Colab Notebooks"
date:   2021-10-31 10:00:00
categories: Education Protein ML
excerpt_separator: <!--more-->
---

Seven Google Colab notebooks made for the [CSBERG](https://csberg.org/) Synthetic Biology course. Content delivered in Summer 2021.

<!--more-->

The table of contents Colab notebook is [here](https://colab.research.google.com/drive/1nH5Zjk2dGBfyZ9yYCIo_ly2sixzc-4IM?usp=sharing).

### 1. Introduction

 - Basic `numpy` and `pytorch` vectorized operations
 - `.backward()`, `.grad`, manual gradient optimization
 - Model saving and loading
 - Curse of dimensionality exercise
 - Loading `.csv` and `.fasta` files of sequences, one-hot encoding
 - PyTorch `Dataset` and `DataLoader`

<script src="https://gist.github.com/tianyu-lu/faf802a58b996061034fcf08f28d1502.js"></script>

### 2. Discriminative Models

 - Two layer fully-connected neural network for catalytic activity prediction
 - Rough Mount Fuji model

<script src="https://gist.github.com/tianyu-lu/4e445cef3262055dca5422bc9cc60cff.js"></script>

### 3. Generative Models

 - Representing multiple sequence alignments as matrices
 - Variational Auto-Encoders trained on Pfam aligned sequences
 - Sampling sequences from VAEs and visualizing results with sequence logos

<script src="https://gist.github.com/tianyu-lu/d7043fa72ad0bbfa2675ad862baefcc0.js"></script>

### 4. Model-based Optimization

 - Latent space optimization
 - Conditioning by Adaptive Sampling ([CbAS](https://arxiv.org/abs/1901.10060))

<script src="https://gist.github.com/tianyu-lu/5f09a0c17f2f6fa87bda000c6e2ebf7b.js"></script>

### 5. Inductive Bias

 - Potts model implementation in PyTorch
 - Attention (WIP) and `nn.Embedding`

<script src="https://gist.github.com/tianyu-lu/a3c84f549b52fe01587c254869e10560.js"></script>

### 6. Language Models

 - `bio_embeddings`
 - Exploratory code to benchmark random embeddings for protein property prediction

<script src="https://gist.github.com/tianyu-lu/0ae826307a7af88665a797f0d3cc05ef.js"></script>

### 7. Structure-based Models

 - `py3Dmol` for visualizing structures in Colab
 - Distance matrix, orientograms from `trRosetta`
 - Molecular dynamics with `OpenMM`

<script src="https://gist.github.com/tianyu-lu/c34dade1ed377b728b092bf92905f0e3.js"></script>

### Slides

The accompanying slides to notebooks 1, 2 and 3. Slides 1-28 can be delivered in about 2 hours.

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTUeqZVRnV9ve020al6VjNDBohMKk_gsFKEagjnYmsUVYFOSnELRpcryHdu0ngX_w/embed?start=false&loop=false&delayms=3000" frameborder="0" width="640" height="389" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>