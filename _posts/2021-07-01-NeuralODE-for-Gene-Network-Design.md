---
layout: post
title:  "NeuralODE for Gene Network Design"
date:   2021-07-01 10:00:00
categories: Research Dynamics ML
excerpt_separator: <!--more-->
---

Colab notebook for designing gene regulatory networks by specifying desired dynamics then backproping through an ODE solver.

<!--more-->

The first section of the notebook is a PyTorch implementation of the ideas presented in [Hiscock 2019](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2788-3), where we can use autograd packages to backpropagate through the steps of an ODE solver. Data from two toy systems are given: the French Flag circuit and a the classic repressilator. The learned parameters are the interactions between species in our system, in this case regulatory genes and their protein products. The beauty of this approach is that it is flexible in the amount of prior knowledge / inductive bias the designer wishes to place on the gene regulatory network. In addition, not all species in the system need to be observed: i.e. there can be unobserved nodes, such as those components of a system not tracked by a fluorescent marker but still critical for generating its dynamical properties. It can identify both the node-node connections of a dynamical network and the weights of each interaction, e.g. whether they are activating or inhibitory. As a proof-of-concept, we show it's possible to backprop through a Fast-Fourier Transform such that the designer can simply specify a desired frequency of a repressilator system and the model will fit the interaction parameters which will achieve that, without having to specify the a full set of time vs. concentration data points. 

The second section shows how we can use a data-driven approach to fit ODE model parameters in Julia using its `DiffEqFlux` package. Training data is from a two-component model of neuron spiking activity, the FitzHugh-Nagumo model. We ask whether increasing levels of inductive bias can improve model generalization to unobserved values of the external current applied. Case one is a fully neural network parameterised system of ODEs. Case two is a mixed neural ODE system where only parts of the ODE is parameterized by a neural network while the remaining parts are given in explicit equation form with trainable parameters. Case three has no neural network component; it is completely parameterized with explicit equations.

The third section is a proof-of-concept PyTorch implementation of a differentiable Gillespie Stochastic Simulation Algorithm. The step that is non-differentiable, namely sampling from a categorical distribution to determine the next reaction, is replaced by a differentiable version using the Gumbel-Softmax trick. We show that the gradient information is indeed useful to change the dynamics of the stochastic Lotka-Volterra model of the oscillation frequency of bunny and wolf populations.

<script src="https://gist.github.com/tianyu-lu/6eaefc1920ab80350a597d1516aad8fd.js"></script>
