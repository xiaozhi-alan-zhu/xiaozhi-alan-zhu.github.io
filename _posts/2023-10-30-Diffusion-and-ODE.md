---
layout: distill
title: [WIP]Notes on diffusion model (III) -- ODE-based diffusion model
date: 2023-10-30
description: DDPM, DDIM
tag: ML
bibliography: reference.bib

toc:
    - name: Introduction
    - name: Denoising Diffusion Probabilistic Models (DDPM)
    - name: Denoising Diffusion Implicit Models (DDIM)
---
# Introduction
The 

# Probability-Flow ODE(PF-ODE)

# Consistency model (CM)

# Poisson Flow Generative Models(PFGM)

If we take a look at DDPM and DDIM, the trained neural network outputs a direction for a given input $$(x_t, t)$$ which means the status at a given time $$t$$.
For these diffusion based model, we expect the neural network can approximate the reverse diffusive steps.
The recent paper () provided a different framework to formulate the process.
Inspired by the gravitivity(or any physical quantity that follows [inverse-square law](https://en.wikipedia.org/wiki/Inverse-square_law)), each example in the dataset can be view as a mass point(or particle) in the space and the entire dataset formed a gravity field.
Intuitively, the generation process can be viewed as a random point, sampled far away from the dataset, moving within the gravity field and falls into some equilibrium point.

## Formulation
In this setup, there is no explicit "forward process" and we can start from the backward process
