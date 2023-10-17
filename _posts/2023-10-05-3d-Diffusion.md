---
layout: distill
title: Diffusion model in 3D space
date: 2023-10-05
description: PVD, DPM, LION, SDFusion
tag: ML
bibliography: reference.bib
---

# Introduction
Diffusion model has proven to be a very powerful generative model and achieved huge success in image synthesis.
One of the most widely known application is the synthesis of images condition on text or image input.
Models like ... take one step further to generate short videos based on instruction.

Meanwhile extending this power model to 3D is far from trivial.
It can be represented in point, voxel, surface and etc.
Unlike image data, all these forms are less structured and should be invariant w.r.t. permutation.
Each of the above data structures requires different techniques to deal with(i.e. PointNet for point cloud, graph neural network for surface data etc).

# Point-Voxel diffusion
Direct application of diffusion models on either voxel and point representation results in poor generation quality.
The paper <d-cite key="zhou20213d"></d-cite> proposed Point-Voxel diffusion(PVD) that utilize both point-wise and voxel-wise information to guide diffusion.
The framework is almost exactly the same as DDPM except for the choice of backbone network.
Instead of using U-Net to denoise, the paper applied point-voxel CNN<d-cite key="liu2019point"></d-cite> to parameterize their model.
Point-voxel CNN is capable of extract features from point cloud data and requires much less computational resource than previous methods(i.e. PointNet).

## Quick review of DDPM
Here, we quickly review the training and sampling process with diffusion model.
The DDPM parameterize the model learn the "noise" $$\epsilon_\theta(x_t, t)$$ and the paper used a point-voxel CNN to represent $$\epsilon_\theta(x_t, t)$$.
The loss function is

$$
\begin{align}
\min_{x_t, \epsilon\sim\mathcal{N}(0, I)}&\|\epsilon-\epsilon_\theta(x_t, t)\|^2_2\\
\text{where } x_t =& \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon.
\end{align}
$$

WIth a trained model $$\epsilon_\theta(x_t, t)$$, the sampling process is
$$
\begin{equation}
\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\tilde{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right)+\sqrt{\beta_t} \mathbf{z},
\end{equation}
$$
where $$z\sim \mathcal{N}(o, I)$$.

In the context of this paper, $$x_0\in\mathbb{R}^{N\times 3}$$ is point cloud data.

## Experiment
In the experiment part, the paper tested the algorithm on tasks of shape generation and shape completeion.
It examined metrics such as [Chamfer Distance(CD)](https://pdal.io/en/latest/apps/chamfer.html), [Earth Mover's Distance(EMD)](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) and etc.

<figure>
  <img src="../../../assets/img/diffusion-model/PVD-generation-samples.png" style="width:100%">
  <figcaption>Fig.1 - Comparison of samples generated from different methods. We can observe that Voxel based diffusion is not enough to generate high quality samples.</figcaption>
</figure>


# Diffusion Probabilistic Models for 3D Point Cloud Generation(DPM)

The paper<d-cite key="luo2021diffusion"></d-cite> proposed a framework that introduced latent encoding to the DDPM.
The diffusion process and reverse process still happens in physical space, not in latent space.
The latent variable is used as a guidence during the reverse process.

For clarity of discussion, we denote $$\boldsymbol{X}=\{x_i\}_{i\in[N]}$$ to be a 3D shape in the form of point cloud and $$x_i\in \mathbb{R}^3$$ for $$i\in [N]$$ represents each point in the point cloud.
Let $$\tilde{p}(\boldsymbol{X})$$ be the distribution of shapes $$\boldsymbol{X}$$ and $$p(x)$$ be the distribution of points in arbitrary shapes.

## Formulation
The paper introduced a latent variable $$\boldsymbol{z}\sim q_\varphi(\boldsymbol{z}\mid\boldsymbol{X}^{(0)})=\mathcal{N}\left(z\mid\mu_\varphi(\boldsymbol{X}^{(0)}, \Sigma_\varphi(\boldsymbol{X}^{(0)}))\right)$$.
However, instead of apply diffusion process in latent space as in <d-cite key="rombach2022high"></d-cite>, it is used only as a guidence during the diffusion $$\boldsymbol{X}^{(t-1)}\sim p_\theta(\cdot\mid\boldsymbol{X}^{(t)}, \boldsymbol{z})$$.
More importantly, unlike PVD or image-based diffusion model, the diffusion process is conducted pointwisely.
To be more specific, each point in a point cloud is diffused separately $$x_i^{(t-1)}\sim p_\theta\left(\cdot\mid x_i^{(t)}, z\right)$$.
The paper further concluded that points in a given point cloud is conditionally independent(given the point cloud/shape) and, mathematically, can be formulated as
$$
\begin{equation}
\tilde{p}_\theta(\boldsymbol{X}\mid\boldsymbol{z})=\prod_{i=1}^N p_\theta(x_i\mid\boldsymbol{z}).
\end{equation}
$$

This is also shown in the following graphical model.
<figure>
  <img src="../../../assets/img/diffusion-model/DPM-graphical-model.png" style="width:100%">
  <figcaption>Fig.2 - The graphical model of the Diffusion Probabilistic Model. The latent embedding is used as a stand alone information. The start point of the diffusion is still Gaussian distributed points in 3D physical space.</figcaption>
</figure>

In comparison, the PVD update each reverse diffusion step through updating all points simutaniously and can be viewed as following graphical model.
<figure>
  <img src="../../../assets/img/diffusion-model/PVD-graphical-model.png" style="width:30%">
  <figcaption>Fig.3 - The graphical model of the PVD. In PVD's algorithm, all points in point cloud is input information for updating a diffusion step.</figcaption>
</figure>

The forward process is the same as DDPM or PVD which is simply adding noise to the points.
The reverse process is formulated as

$$
\begin{gathered}
p_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{(0: T)} \mid \boldsymbol{z}\right)=p\left(\boldsymbol{x}^{(T)}\right) \prod_{t=1}^T p_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{(t-1)} \mid \boldsymbol{x}^{(t)}, \boldsymbol{z}\right), \\
p_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{(t-1)} \mid \boldsymbol{x}^{(t)}, \boldsymbol{z}\right)=\mathcal{N}\left(\boldsymbol{x}^{(t-1)} \mid \boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{(t)}, t, \boldsymbol{z}\right), \beta_t \boldsymbol{I}\right)。
\end{gathered}
$$

In this setup, $$p_\theta(x^{(0)})$$ represent an image's probability under the learned model $$p_\theta$$ and, in below, we will use $$q_\varphi(z\mid \boldsymbol{X}^{(0)})$$ as the encoder.
For simplicity of discussion, we will use $$p_\theta$$ to replace $$\tilde{p}_\theta$$.


## Training objective
Similar to DDPM, the log-likelihood of reverse process $$p_\theta(x^{(0)})$$ can be lower bounded by

$$
\begin{aligned}
\mathbb{E}\left[\log p_{\boldsymbol{\theta}}\left(\boldsymbol{X}^{(0)}\right)\right] &\geq \mathbb{E}_q  {\left[\log \frac{p_{\boldsymbol{\theta}}\left(\boldsymbol{X}^{(0: T)}, \boldsymbol{z}\right)}{q\left(\boldsymbol{X}^{(1: T)}, \boldsymbol{z} \mid \boldsymbol{X}^{(0)}\right)}\right] } \\
&=\mathbb{E}_q\left[\log p\left(\boldsymbol{X}^{(T)}\right)\right.  \\
&~~~~~~ +\sum_{t=1}^T \log \frac{p_{\boldsymbol{\theta}}\left(\boldsymbol{X}^{(t-1)} \mid \boldsymbol{X}^{(t)}, \boldsymbol{z}\right)}{q\left(\boldsymbol{X}^{(t)} \mid \boldsymbol{X}^{(t-1)}\right)} \\
&~~~~~~ \left.-\log \frac{q_{\boldsymbol{\varphi}}\left(\boldsymbol{z} \mid \boldsymbol{X}^{(0)}\right)}{p(\boldsymbol{z})}\right].
\end{aligned}
$$

The process is parameterized by $$\theta, \varphi$$ and the variational lower bound can be written into KL divergencies,

$$
\begin{gathered}
L(\boldsymbol{\theta}, \boldsymbol{\varphi})=\mathbb{E}_q\left[\sum _ { t = 2 } ^ { T } D _ { \mathrm { KL } } \left(q\left(\boldsymbol{X}^{(t-1)} \mid \boldsymbol{X}^{(t)}, \boldsymbol{X}^{(0)}\right) \|\right.\right. \\
\left.p_{\boldsymbol{\theta}}\left(\boldsymbol{X}^{(t-1)} \mid \boldsymbol{X}^{(t)}, \boldsymbol{z}\right)\right) \\
-\log p_{\boldsymbol{\theta}}\left(\boldsymbol{X}^{(0)} \mid \boldsymbol{X}^{(1)}, \boldsymbol{z}\right) \\
\left.+D_{\mathrm{KL}}\left(q_{\boldsymbol{\varphi}}\left(\boldsymbol{z} \mid \boldsymbol{X}^{(0)}\right) \| p(\boldsymbol{z})\right)\right] .
\end{gathered}
$$

This objective function can be then written in the form of points(instead of shapes) as follows,

$$
\begin{aligned}
L(\theta, \varphi) &= \mathbb{E}_q \Big[\sum_{t=2}^T\sum_{i=1}^N D_\mathrm{KL}\left(q(x_i^{(t-1)}\mid x_i^{(t)}, x_i^{(0)})\| p_\theta(x_i^{(t-1)}\mid x_i^{(t)}, z)\right)\\
&~~~~~~~~ -\sum_{i=1}^N\log p_\theta(x_i^{(0)}\mid x_i^{(1)}, z)+D_\mathrm{KL}(q_\varphi(z\mid \boldsymbol{X}^{(0)})\| p(z))\Big].
\end{aligned}
$$

Another important point about this paper is that it does not assume the prior distribution of the latent variable $$z$$ to be standard normal.
Instead, the algorithm needs to learn the prior distribution for sampling.
Therefore, both side of $$D_{\mathrm{KL}}\left(q_{\boldsymbol{\varphi}}\left(\boldsymbol{z} \mid \boldsymbol{X}^{(0)}\right) \| p(\boldsymbol{z})\right)$$ involves trainable parameters.
An encoder $$q_{\boldsymbol{\varphi}}\left(\boldsymbol{z} \mid \boldsymbol{X}^{(0)}\right)$$ is learned by parameterizing it with $$\boldsymbol{\mu}_{\varphi}\left(\boldsymbol{X}^{(0)}\right), \boldsymbol{\Sigma}_{\boldsymbol{\varphi}}\left(\boldsymbol{X}^{(0)}\right)$$.
An map $$F_\alpha$$ that transform samples in standard normal to prior distribution is learned by parameterizing it with a bijection neural network and $$z=F_\alpha(\omega)$$.

<figure>
  <img src="../../../assets/img/diffusion-model/DPM-flowchart.png" style="width:100%">
  <figcaption>Fig.4 - The flowchart of training and sampling process of DPM.</figcaption>
</figure>

<figure>
  <img src="../../../assets/img/diffusion-model/DPM-algorithm.png" style="width:100%">
  <figcaption>Fig.5 - DPM training algorithm.</figcaption>
</figure>

# Latent Point Diffusion Models(LION)
The latent diffusion model<d-cite key="rombach2022high"></d-cite> in image synthesis conducted the diffusion process in the latent space.
The paper<d-cite key="vahdat2022lion"></d-cite>, unlike previously mentioned PVD<d-cite key="luo2021diffusion"></d-cite>, applied the similar spirit to the 3D point clouds.

## Formulation
The model is consist of a VAE to encode shapes to latent space and diffusion models that map vectors from standard normal distribution to latent space.

**Stage 1: VAE**

The VAE part, the encoding-decoding process has three steps:
1. Use a PVCNN to encode the whole point clouds into a latent vector (shape latent) $$z_0\in \mathbb{R}^{D_z}$$.
2. Concatenate shape latent with each point in the point cloud. Then use a PVCNN to map point clouds to latent "point clouds" $$h_0\in \mathbb{R}^{3+D_h}$$ in latent space.
3. Decoding from the concatenation of latent points and shape latent.
<figure>
  <img src="../../../assets/img/diffusion-model/lion-stage-1-flowchart.png" class="center" style="height:90%">
  <figcaption>Fig.6 - LION training-Stage 1: VAE part.</figcaption>
</figure>

**Stage 2: Diffusion**

There are two diffusion processes involved since there are two latent vectors.
Both diffusion processes start from standard normal distribution and mapped to shape latent vectors and point latent respectively.
<figure>
  <img src="../../../assets/img/diffusion-model/lion-stage-2-flowchart.png" class="center" style="height:90%">
  <figcaption>Fig.6 - LION training-Stage 1: VAE part.</figcaption>
</figure>

**Sampling process**

The sample generation process is consist of three steps:
1. Sample a vector from multivariate standard normal $$z_T$$ distribution and reverse diffused into shape latent $$z_0$$.
2. Sample a vector from multivariate standard normal $$h_T$$ and concatenate shape latent $$z_0$$ with each intermediate step $$h_t$$ and reversely diffused into point latents.

## Training objective
During the training of VAE, LION is trained by maximizing a modified variational lower bound on the data log-likelihood with respect to the encoder and ecoder parameters $$\phi$$ and $$\xi$$:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}}(\boldsymbol{\phi}, \boldsymbol{\xi}) & =\mathbb{E}_{p(\mathbf{x}), q_\phi\left(\mathbf{z}_0 \mid \mathbf{x}\right), q_\phi\left(\mathbf{h}_0 \mid \mathbf{x}, \mathbf{z}_0\right)}\left[\log p_{\boldsymbol{\xi}}\left(\mathbf{x} \mid \mathbf{h}_0, \mathbf{z}_0\right)\right. \\
& \left.-\lambda_{\mathbf{z}} D_{\mathrm{KL}}\left(q_\phi\left(\mathbf{z}_0 \mid \mathbf{x}\right) \mid p\left(\mathbf{z}_0\right)\right)-\lambda_{\mathbf{h}} D_{\mathrm{KL}}\left(q_\phi\left(\mathbf{h}_0 \mid \mathbf{x}, \mathbf{z}_0\right) \mid p\left(\mathbf{h}_0\right)\right)\right] .
\end{aligned}
$$

The priors $$p(z_0)$$ and $$p(h_0)$$ are $$\mathcal{N}(o, I)$$.
During the training of diffusion models, the models are trained on embeddings and have VAE model fixed.

# 3D-LDM


# SDFusion

