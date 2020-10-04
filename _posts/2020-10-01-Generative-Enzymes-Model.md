---
layout: post
title:  "Generative-Enzymes-Model"
date:   2020-10-01 10:00:00
categories: Research Protein ML
excerpt_separator: <!--more-->
---

Adapting the CbAS algorithm to generate plastic degrading enzymes with information from structure.

<!--more-->

Animation of the <a href="https://arxiv.org/abs/1901.10060">CbAS</a> algorithm by <a href="https://yinqihuang.myportfolio.com/">Yinqi Huang</a>:

<img src="/assets/images/cbas_smooth.gif">

A VAE is trained over existing enzyme sequences to parameterize a prior distribution. Feed forward neural networks are trained to predict an enzyme's catalytic activity and thermostability from its sequence. At generation time, 1000 Gaussian normal samples are drawn from the VAE's latent space and decoded to generate novel protein sequences. Each one is scored by the feed forward neural networks to predict function. Samples are reweighed, with more weight going to samples with high predicted function and closeness to the prior. The VAE is retrained on the reweighed data for the next iteration.

<a href="https://hackmd.io/yQ5gq3cMSxuWYFBw472h8g?view">Writeup</a>

<a href="https://colab.research.google.com/drive/1YUHtklaLW1LlCSHCYiE0PrTj85vzNLwa?usp=sharing">Code</a>