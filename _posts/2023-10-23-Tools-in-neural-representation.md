---
layout: distill
title: [WIP]Neural representation for geometric learning
date: 2023-10-23
description: 
tag: ML
bibliography: reference.bib
toc: true
toc_depth: 2
---

# NeRF (Neural radiance fields)
The problem NeRF aims to solve is the view synthesis, that is given a dense sampling of view, how to generate photorealistic novel views.
The way NeRF did is to build a neural network that can output images of observation for an input of an location $$(x, y, z)$$ and an direction $$(\theta, \phi)$$.
In <d-cite key="mildenhall2021nerf"></d-cite>, the author proposed to use fully-connected neural network(without any convolutional layers) and generate a single volume density and view-dependent RGB color.

## Formulation of Neural Radiance Field Scene Representation
The paper <d-cite key="mildenhall2021nerf"></d-cite> approximate the scene by constructing an continuous 5D MLP network $$F_\Theta: (x, d)\rightarrow(c, \sigma)$$ and optimize its weights $$\Theta$$ to map from each input 5D coordinate to its corresponding volume density and directional emitted color.
In this function $$F_\Theta$$, the input is 3D location $$\mathbb{x}=(x,y,z)$$ and 2D viewing direction $$(\theta, \phi)$$,  and whose output is an emitted color $$c=(r, g, b)$$ and volume density $$\sigma$$.
In practice, the volume density $$\sigma(\mathbf{x})$$ can be interpreted as the differential probability of a ray terminating at an infinitesimal particle at location $$\mathbf{x}$$.
The expected color $$C(\mathbf{r})$$ of camera ray $$\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$$ with near and far bounds $$t_n$$ and $$t_f$$ is:

$$
\begin{equation}
C(\mathbf{r})=\int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) d t, \text { where } T(t)=\exp \left(-\int_{t_n}^t \sigma(\mathbf{r}(s)) d s\right).
\end{equation}
$$

Numerically, this continuous integral is approximated by using quadrature.
Stratified sampling approach has been used to avoid deterministicly query fixed set of locations, which limited representation's resolution.
To be more specific, the interval $$[t_n, t_f]$$ is partitioned into $$N$$ evenly-spaced bins and uniformly draw one sample from each bin:

$$
t_i \sim \mathcal{U}\left[t_n+\frac{i-1}{N}\left(t_f-t_n\right), t_n+\frac{i}{N}\left(t_f-t_n\right)\right].
$$

These samples are then used to estimate $$C(\mathbf{r})$$ as follows,

$$
\hat{C}(\mathbf{r})=\sum_{i=1}^N T_i\left(1-\exp \left(-\sigma_i \delta_i\right)\right) \mathbf{c}_i, \text { where } T_i=\exp \left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right),
$$

where $$\delta_i=t_{i+1}-t_i$$ is the distance between adjacent samples.

In the implementation of this algorithm, there are few more components:
1. Positional encoding:
Deep networks are biased towards learning lower frequenecy functions <d-cite key="rahaman2019spectral"></d-cite>.
One way to improve is mapping the inputs to a higher dimensional space using high frequency functions before passing them to the network.
In this spirit, the paper reformulated $$F_\Theta$$ as a composition of two functions $$F_\Theta=F'_\Theta\circ\gamma$$, where $$\gamma$$ is defined as
$$
\begin{equation}
\gamma(p)=\left(\sin \left(2^0 \pi p\right), \cos \left(2^0 \pi p\right), \cdots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right).
\end{equation}
$$
The paper claimed that this process greatly improved model's performance.

2. Hierarchical volume sampling:
Evenly sampling along camera ray is relatively inefficient, since free space and occluded regions do not contribute to the rendered image but still sampled repeatedly.
The paper proposed an two stages method.
First train a "coarse" network $$\hat{C}_c(\mathbf{r})$$ that sample a set of $$N_c$$ locations using stratified sampling and train on it.
Then, given the output of this "coarse" network, produce a more informed sampling of points along each ray $$N_f$$.
Finally, using all $$N_c+N_f$$ samples to train a "finer" network $$\hat{C}_f(\mathbf{r})$$.


# SIREN (Sinusoidal Representation Networks)
SIREN is proposed in <d-cite key="sitzmann2020implicit"></d-cite>.
It is a simple neural network architecture for implicit neural representations that uses the sine as a periodic activation function:

$$
\Phi(\mathbf{x})=\mathbf{W}_n\left(\phi_{n-1} \circ \phi_{n-2} \circ \ldots \circ \phi_0\right)(\mathbf{x})+\mathbf{b}_n, \quad \mathbf{x}_i \mapsto \phi_i\left(\mathbf{x}_i\right)=\sin \left(\mathbf{W}_i \mathbf{x}_i+\mathbf{b}_i\right) .
$$

Here, $$\phi_i:\mathbb{R}^{M_i}\rightarrow \mathbb{R}^{N_i}$$ is the $$i^\mathrm{th}$$ layer of the network.
An important feature of SIREN is that the gradient of a SIREN $$\Phi(x)$$ with regards to its input $$x$$ is still a SIREN.
It is demonstrated in the appendix section 2 of <d-cite key="sitzmann2020implicit"></d-cite> that the gradient operator will shift the phase of the sine function by $$\frac{\pi}{2}$$.
