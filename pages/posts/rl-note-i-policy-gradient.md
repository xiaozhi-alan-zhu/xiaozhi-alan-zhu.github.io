---
title: Reinforcement Learning Notes (I) -- Policy Gradient
description: Minimum introduction to policy gradient
date: 2025-12-21
tags: [reinforcement-learning, policy-gradient]
---

This is the first post of a series of notes on reinforcement learning. 
The aim for this series is to document my understanding of RL and provide a minimal level of knowledge (but with reasonable amount of mathematical rigor) for engineers to get started with RL.

## 1. The Goal of Reinforcement Learning

The goal is to learn a policy $\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)$ that maximizes the expected return $J(\theta)$.

$$
J(\theta) = E_{\tau \sim \pi_\theta}[r(\tau)] = \int p_\theta(\tau) r(\tau) d\tau
$$

Where $\tau$ is a trajectory $(\mathbf{s}_1, \mathbf{a}_1, \dots, \mathbf{s}_T, \mathbf{a}_T)$ and $p_\theta(\tau)$ is the probability of the trajectory under policy $\pi_\theta$:

$$
p_\theta(\tau) = p(\mathbf{s}_1) \prod_{t=1}^T \pi_\theta(\mathbf{a}_t|\mathbf{s}_t) p(\mathbf{s}_{t+1}|\mathbf{s}_t, \mathbf{a}_t)
$$

Where the notation is defined as follows:

*   $\mathbf{s}_t$: State at time step $t$.
*   $\mathbf{a}_t$: Action at time step $t$.
*   $r(\mathbf{s}_t, \mathbf{a}_t)$: Reward function.
*   $\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)$: Policy with parameters $\theta$.
*   $p(\mathbf{s}_{t+1} | \mathbf{s}_t, \mathbf{a}_t)$: Transition dynamics of the environment.
*   $p(\mathbf{s}_1)$: Initial state distribution.
*   $\tau$: A trajectory sequence $\{\mathbf{s}_1, \mathbf{a}_1, \dots, \mathbf{s}_T, \mathbf{a}_T\}$.
*   $p_\theta(\tau)$: Probability of observing trajectory $\tau$ under policy $\theta$.
*   $r(\tau)$: Cumulative return of trajectory $\tau$.

The following diagram illustrates the relationship between these variables in a Markov Decision Process (MDP):

![MDP Diagram](/assets/mdp_diagram.png)


## 2. The Policy Gradient
We want to update $\theta$ in the policy with $\nabla_\theta J(\theta)$. Using the **log-derivative trick**: $\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$, we get:

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) r(\tau) d\tau = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) r(\tau) d\tau
$$

$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} [\nabla_\theta \log p_\theta(\tau) r(\tau)]
$$

Expanding $\log p_\theta(\tau)$, the terms involving the dynamics $p(\mathbf{s}_{t+1}|\mathbf{s}_t, \mathbf{a}_t)$ do not depend on $\theta$, so their gradient is zero. We are left with:

$$
\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \underbrace{\left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t) \right)}_{\substack{\text{gradient of policy} \\ \text{over trajectory}}} \quad \underbrace{\left( \sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) \right)}_{\substack{\text{cumulative reward} \\ \text{of trajectory}}} \right]
$$

### Interpretation
This equation decomposes the gradient into two parts:
1.  **Policy Direction**: $\sum \nabla_\theta \log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)$ represents the direction in parameter space that increases the probability of the actions taken in the trajectory.
2.  **Trajectory Reweighting**: $\sum r(\mathbf{s}_t, \mathbf{a}_t)$ acts as a scalar weight.

The gradient update pushes the policy parameters $\theta$ in the direction of trajectories that yield high cumulative rewards ("trial and error"). Formally, it scales the gradient of the log-probability of the trajectory by its return.

### The REINFORCE Algorithm

The REINFORCE algorithm (Williams, 1992) is the simplest implementation of the policy gradient. It uses Monte Carlo sampling to estimate the return $r(\tau)$.

$$
\begin{array}{l}
\hline
\textbf{Algorithm 1} \text{ REINFORCE Algorithm} \\
\hline
\textbf{Input: } \text{differentiable policy } \pi_\theta, \text{ learning rate } \alpha \\
\textbf{Initialize: } \text{parameters } \theta \text{ at random} \\
\textbf{for } \text{each episode } k = 1, \dots, M \textbf{ do} \\
\quad \text{Generate trajectory } \tau = (\mathbf{s}_1, \mathbf{a}_1, r_1, \dots, \mathbf{s}_T, \mathbf{a}_T, r_T) \sim \pi_\theta \\
\quad \text{Compute gradient } \nabla J(\theta) \approx \sum_i(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)) (\sum_{t=1}^T r_{t}) \\
\quad \text{Update } \theta \leftarrow \theta + \alpha \nabla J(\theta) \\
\textbf{end for} \\
\hline
\end{array}
$$

Since it relies on the full trajectory return (Monte Carlo), REINFORCE is unbiased but suffers from **high variance**, often leading to slow convergence.

**Why is variance high?**
1.  **Stochasticity**: In a standard MDP, both the policy and the environment are stochastic. A single sampled trajectory is just one realization of a highly variable process. High-probability paths might yield low rewards due to a few unlucky transitions, and vice-versa.
2.  **Difficulty in Credit Assignment**: REINFORCE uses the *total* return $G_t$ to update *all* actions in the trajectory. If a trajectory has a high return, the algorithm reinforces *every* action taken, even if some were suboptimal. Without a critic to evaluate individual states, the signal ("good" or "bad") is smeared across the entire sequence, introducing significant noise.
3.  **Magnitude of Returns**: The gradient updates are scaled by the return $G_t$. If returns vary wildly in magnitude (e.g., one path gives 0, another gives 1000), the gradient updates will swing violently, destabilizing the learning process.

## 3. Variance Reduction

The standard policy gradient estimator has high variance. We can reduce this variance using two main techniques: **Causality** and **Baselines**.

### 3.1 Exploiting Causality (Reward-to-Go)
The policy at time $t$ cannot affect rewards obtained in the past ($t' < t$). Therefore, we can replace the total return with the **reward-to-go**:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t}) \left( \sum_{t'=t}^T r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)
$$

The term $\sum_{t'=t}^T r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'})$ is often denoted as $\hat{Q}_{i,t}$.

### 3.2 Baselines
We can subtract a baseline $b(\mathbf{s}_t)$ from the return without introducing bias, as long as the baseline does not depend on the action $\mathbf{a}_t$:

$$
E [\nabla_\theta \log \pi_\theta(\mathbf{a}_t|\mathbf{s}_t) b(\mathbf{s}_t)] = 0
$$

Thus, the policy gradient with a baseline is:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t}) \left( \hat{Q}_{i,t} - b(\mathbf{s}_{i,t}) \right)
$$

A common choice for the baseline is the average reward or a learned Value Function $V_\phi(\mathbf{s}_t)$. This significantly reduces variance.

## 4. Implementation

In automatic differentiation frameworks (like PyTorch or TensorFlow), we don't compute the gradient manually. Instead, we construct a "surrogate loss" whose gradient equals the policy gradient.

$$
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t}) \hat{A}_{i,t}
$$

Where $\hat{A}_{i,t}$ is the estimated advantage (e.g., $\hat{Q}_{i,t} - b(\mathbf{s}_{i,t})$). We treat $\hat{A}_{i,t}$ as a fixed constant (detach from graph) during backpropagation.

**Algorithm:**
1.  **Sample**: Run policy $\pi_\theta$ to collect trajectories $\{\tau_i\}$.
2.  **Estimate Return**: Compute reward-to-go $\hat{Q}_{i,t}$ and optionally fit a baseline $V_\phi$.
3.  **Update**: Compute gradient $\nabla_\theta J$ and update $\theta \leftarrow \theta + \alpha \nabla_\theta J$.


### Suggested Readings
*   **Williams (1992)**: Simple statistical gradient-following algorithms (REINFORCE).
*   **Sutton et al. (2000)**: Policy Gradient Theorem.
*   **Schulman et al. (2015)**: Trust Region Policy Optimization (TRPO).
