---
title: DINO Series
description: A detailed summary of DINO models series (v1, v2, v3)
date: 2025-12-27
tags: [DINO, ViT, self-supervised-learning]
# draft: true
---

## 1. Introduction: The Self-Supervised Revolution

> "Self-supervised learning is the cake, supervised learning is the icing on the cake, and reinforcement learning is the cherry on the cake." — Yann LeCun

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="/assets/lecun-cake.png" alt="The 'Cake' Analogy" style="width: 80%;" />
  <p style="text-align: center; color: #555; font-style: italic; margin-top: 8px;">The "Cake" Analogy: SSL provides dense feedback signal (millions of bits) compared to sparse rewards in RL.</p>
</div>



### 1.1 The Context: The "Data Hungry" Challenge
When **Vision Transformer (ViT)** was first introduced [@dosovitskiy2020image], it faced three hurdles preventing it from simply replacing CNNs:

1.  **Lack of Inductive Bias**: CNNs have built-in "assumptions" about images (locality, shift invariance). ViTs, treating images as generic sequences, must *learn* these spatial rules from scratch, making them extremely **data-hungry**.
2.  **Rugged Optimization Landscape**: Without these priors, ViTs are notoriously hard to optimize. Training from scratch on standard datasets (like ImageNet-1k) often leads to overfitting or unstable convergence compared to ResNets.
3.  **Signal Sparsity**: Supervised Learning gives a "weak" signal (one integer label per image). For a model as flexible as ViT, this scalar feedback is insufficient to constrain the massive parameter space effectively. They need "dense" supervision—which is exactly what Self-Supervised Learning (SSL) provides.


**Enter Self-Supervised Learning (SSL)**:
As Yann LeCun's "Cake" analogy suggests, supervised learning (labels) limits the information signal. SSL provides a dense signal that allows ViT to learn these structural rules from the data itself, without needing 300M human labels.

Before DINO, SSL was dominated by **Contrastive Learning**:
*   **Contrastive Learning**: Push positive pairs (views of same image) together, push negative pairs (different images) apart.
*   **The Limitation**: Requires large batch sizes or memory banks for negative samples to be effective.

**Distillation approaches** (like BYOL) emerged to remove the need for negative pairs, relying only on positive pairs. DINO (2021) took this further by integrating it with **Vision Transformers (ViTs)**.

### 1.2 The Genealogy of DINO
*   **DINO (2021)**: "Emerging Properties in Self-Supervised Vision Transformers". Proved ViTs can learn unsupervised segmentations.
*   **DINOv2 (2023)**: "Learning Robust Visual Features without Supervision". Scaling up data and combining objectives (iBOT) for a universal backbone.
*   **DINOv3 (2025)**: "A Vision Transformer for Every Use Case". Solving the "dense feature degradation" problem in long training runs using **Gram Anchoring**.

---

## 2. DINO (v1): Self-Distillation with NO Labels

> **Core Idea**: If we treat the same network as both a "Teacher" and a "Student", can the student learn from the teacher's output without collapsing to a trivial solution?

### 2.1 High-Level Concept

The core idea is **self-distillation**. A **student network** is trained to match the output distribution of a **teacher network**. The key is that the teacher network's weights are not learned through backpropagation; instead, they are a "momentum" version of the student's weights, which provides a more stable learning target.

### 2.2 Network Architecture

- **Student Network ($g_{\theta_s}$):** This is the main network being trained. It consists of a backbone (like a Vision Transformer or ViT) and a projection head (an MLP). Its weights, $\theta_s$, are updated via backpropagation.
    
- **Teacher Network ($g_{\theta_t}$):** This network has the **exact same architecture** as the student. Its weights, $\theta_t$, are _not_ updated by the optimizer. Instead, they are an **exponential moving average (EMA)** of the student's weights.

### 2.3 The DINO Algorithm: Step-by-Step

Here is the process for a single training step:

####  Step 1: Data Preparation (Multi-Crop Augmentation)

From a single input image (e.g., a $1080 \times 1080$ photo), the algorithm generates a "batch" of different views. This batch is split into two main categories:

1. **Global Views (The "Teacher's" View):**
    - **What they are:** Two separate, large crops are taken from the original image.
    - **Process:** The algorithm randomly selects a large area (e.g., 50% to 100% of the original image) and a random aspect ratio. This crop is then resized to the network's standard input size (e.g., **$224 \times 224$ pixels**). There will be a meaningful amount of overlap between two images.
    - **Purpose:** These views contain the overall scene and context—the "big picture."
2. **Local Views (The "Student's" Test):**
    - **What they are:** Several (e.g., 4, 6, or 8) additional, small crops are taken.
    - **Process:** The algorithm randomly selects very small areas (e.g., 5% to 40% of the original image). These tiny crops are then resized to a much smaller input size (e.g., **$96 \times 96$ pixels**).
    - **Purpose:** These views act as "zoomed-in" details, like looking at just an eye, a wheel, or a single leaf.



<div style="border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f9f9f9; font-size: 0.8em; width: 45%; float: right; margin-left: 20px; margin-bottom: 20px;">


**The Augmentation Pipeline**

Crucially, **every single crop** (both global and local) is passed through a strong data augmentation pipeline to make the task harder and prevent the model from "cheating" by just memorizing simple colors or textures. This pipeline includes:

- **Random Resized Crop:** This is the multi-crop process itself.
- **Random Horizontal Flip:** Flips the image with a 50% probability.
- **Color Jitter:** Randomly changes the image's **brightness**, **contrast**, **saturation**, and **hue**.
- **Gaussian Blur:** Randomly applies a blur to the image.
- **Solarization:** A strong augmentation that inverts the pixel values above a certain threshold.

</div>


By the end of this step, you have a set of highly varied, distorted images. The student network is then given the difficult task (in Step 5) of proving that a tiny, blurred, solarized $96 \times 96$ local view (like a patch of fur) belongs to the same _global concept_ as the large, $224 \times 224$ global view of the whole animal. This forces it to learn what "fur" is in a general sense, rather than just memorizing a specific patch.


<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="/assets/dino-multicrop.png" alt="DINO Multi-Crop Strategy" style="width: 100%;" />
  <p style="text-align: center; color: #555; font-style: italic; margin-top: 8px;">DINO Multi-Crop Strategy: 2 Global crops (>50%) cover the scene, while multiple Local crops (<50%) capture details. The student must match the teacher's global view from only a local view.</p>
</div>


#### Step 2: Forward Pass


The crops are passed through the networks differently:

- **All crops** (both global and local) are fed into the **student network** $g_{\theta_s}$.
- **Only the two global crops** are fed into the **teacher network** $g_{\theta_t}$.

This "local-to-global" strategy forces the student to learn global information even when it's only looking at a small patch.

#### Step 3: Avoiding Collapse (Centering & Sharpening)

Before calculating the loss, the network outputs are processed to prevent "model collapse" (where the network outputs the same thing for every input).

1. **Teacher Output (Sharpening):** The teacher network's $K$-dimensional output is passed through a `softmax` function with a very low **temperature** $\tau_t$ (e.g., 0.04). This "sharpens" the probability distribution, making it more peaked and preventing collapse to a uniform distribution.

    <div style="display: flex; flex-direction: column; align-items: center; margin: 20px 0;">
      <img src="/assets/softmax-temperature-demo.png" alt="Effect of Temperature on Softmax" style="width: 100%; border: 1px solid #ddd; border-radius: 8px;" />
      <p style="text-align: center; color: #555; font-style: italic; margin-top: 8px;">Effect of Temperature on Softmax: Lower temperature (T=0.1) creates a sharper distribution (peaked), acting as a pseudo-label for the student.</p>
    </div>

    
2. **Teacher Output (Centering):** The teacher's outputs are "centered" by subtracting a moving average $c$ of the outputs from the entire batch. This center $c$ is also updated using an EMA. This prevents a single dimension from dominating the output.
    
3. **Student Output:** The student's $K$-dimensional output is passed through a `softmax` with a higher **temperature** $\tau_s$ (e.g., 0.1).

#### Step 4: Loss Calculation (Cross-Entropy)

The goal is to make the student's output distribution, $P_s$, match the teacher's centered and sharpened output distribution, $P_t$.

This is done by minimizing a **cross-entropy loss**:




$$
L = - \frac{1}{V} \sum_{x \in \{x^g_1, x^g_2, ... x^l_V\}} \left[ P_t(x^g_1) \log P_s(x) + P_t(x^g_2) \log P_s(x) \right]
$$




In plain English:

- For each crop $x$ fed through the student, its output $P_s(x)$ is compared against _both_ global teacher outputs ($P_t(x^g_1)$ and $P_t(x^g_2)$).
- The total loss is the average of all these cross-entropy comparisons.

A **stop-gradient (sg)** is applied to the teacher's output, so the gradient only flows back through the student network.

#### Step 5: Weight Update

Two different updates happen at the end of the step:

1. **Student Weights ($\theta_s$):** The student's weights are updated using backpropagation and an optimizer (like SGD) based on the loss $L$ calculated in the previous step.
    
2. **Teacher Weights** ($\theta_t$): The teacher's weights are updated as an EMA of the student's newly updated weights:
    



    $$
    \theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s
    $$



    
    The $\lambda$ parameter is a momentum schedule that slowly increases from a value like 0.996 up to 1 during training, making the teacher's updates gradually "freeze".

### 2.4 Mathematical Detail: Algorithm Summary


$$

\begin{array}{l}
\hline
\textbf{Algorithm 1: DINO Pseudo-code} \\
\hline
\textbf{input: } \text{Dataset } \mathcal{D}, \text{Student } g_{\theta_s}, \text{Teacher } g_{\theta_t}, \text{Temp } \tau_s, \tau_t, \text{Momentum } \lambda, \text{Center } c \\
\text{Initialize } \theta_s \text{ randomly} \\
\text{Initialize } \theta_t \leftarrow \theta_s \\
\text{Initialize } c \leftarrow 0 \\
\textbf{for } x \text{ in } \mathcal{D} \textbf{ do} \\
\quad \textbf{1. Multi-Crop Augmentation:} \\
\quad \quad V \leftarrow \text{Augment}(x) = \{x_1^g, x_2^g, x_1^l, ..., x_N^l\} \\
\quad \textbf{2. Forward Pass \& Probabilities:} \\
\quad \quad P_s(x') = \text{softmax}(g_{\theta_s}(x') / \tau_s) \quad \forall x' \in V \\
\quad \quad P_t(x^g) = \text{softmax}((g_{\theta_t}(x^g) - c) / \tau_t) \quad \forall x^g \in \{x_1^g, x_2^g\} \\
\quad \textbf{3. Compute Loss:} \\
\quad \quad L = \sum_{x^g} \sum_{x' \neq x^g} - \sum_i P_t(x^g)^{(i)} \log P_s(x')^{(i)} \\
\quad \textbf{4. Update Parameters:} \\
\quad \quad \theta_s \leftarrow \text{Optimizer}(\theta_s, \nabla_{\theta_s} L) \\
\quad \quad \theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s \\
\quad \quad c \leftarrow m c + (1 - m) \frac{1}{B} \sum_{x^g} g_{\theta_t}(x^g) \\
\textbf{end for} \\
\hline
\end{array}

$$




### 2.5 Performance Evaluation

The DINO paper evaluates the model using two primary classification metrics on ImageNet:
1.  **Linear Probe**: 
    *   **Protocol**: Freeze the backbone (weights are fixed). Train a *single linear layer* (Logistic Regression) on top of the features for classification.
    *   **Meaning**: Measures **"Linear Separability"**. It asks: "Are the features arranged simply enough that a straight line can separate cats from dogs?"

2.  **$k$-NN Classifier**: 
    *   **Protocol**: Freeze the backbone. For a test image, compare its feature vector to all training images using **Cosine Similarity**. The class is determined by a vote of the $k=20$ nearest neighbors.
    *   **Meaning**: Measures **"Manifold Structure"**. It asks: "Do images of the same class naturally clump together in the feature space without *any* supervisory training?"

3.  **Throughput (im/s)**: How fast the model processes images (images per second).

#### **Key Findings (from Table 2):**


<div style="float: right; margin-left: 20px; margin-bottom: 20px; font-size: 0.7em; max-width: 50%;">

| Method | Arch. | Param. | im/s | Linear | $k$-NN |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | | | | | |
| Supervised | RN50 | 23 | 1237 | 79.3 | 79.3 |
| SCLR | RN50 | 23 | 1237 | 69.1 | 60.7 |
| MoCov2 | RN50 | 23 | 1237 | 71.1 | 61.9 |
| BYOL | RN50 | 23 | 1237 | 74.4 | 64.8 |
| SwAV | RN50 | 23 | 1237 | **75.3** | 65.7 |
| **DINO** | **RN50** | 23 | 1237 | **75.3** | **67.5** |
| **ViT-Small** | | | | | |
| Supervised | ViT-S | 21 | 1007 | 79.8 | 79.8 |
| BYOL* | ViT-S | 21 | 1007 | 71.4 | 66.6 |
| MoCov2* | ViT-S | 21 | 1007 | 72.7 | 64.4 |
| SwAV* | ViT-S | 21 | 1007 | 73.5 | 66.3 |
| **DINO** | **ViT-S** | 21 | 1007 | **77.0** | **74.5** |
| **Cross-Architecture** | | | | | |
| DINO | ViT-B/16 | 85 | 312 | 78.2 | 76.1 |
| DINO | ViT-S/8 | 21 | 180 | 79.7 | **78.3** |
| DINO | ViT-B/8 | 85 | 63 | **80.1** | 77.4 |

</div>



*   **ViT vs. ResNet**: DINO works best with Vision Transformers. While it matches methods like SwAV on ResNet-50 (75.3%), it significantly outperforms them when using ViT architectures.
*   **The Patch Size Trade-off**:
    *   **ViT-S/16** (Standard 16x16 patches): High speed (**1007 im/s**) with decent accuracy (77.0%).
    *   **ViT-S/8** (Smaller 8x8 patches): Significant accuracy boost (**79.7% Linear**, **78.3% k-NN**) but much slower speed (**180 im/s**).
    *   *Correction on "Batch Size"*: The performance gain comes from smaller **Patch Size** (finer granularity), not smaller Batch Size. Smaller patches typically improve accuracy but quadratically increase computational cost.
*   **$k$-NN Performance**: DINO achieves remarkably high $k$-NN accuracy (78.3% with ViT-S/8), suggesting the embedding space is highly structured and semantically meaningful even without any classifier training.

### 2.6 Emerging Properties
DINO v1 revealed something magical: **Self-Attention maps automatically segment objects**.
*   In supervised training, the CLS token focuses on discriminative parts (e.g., the dog's ear).
*   In DINO, the attention maps cover the *entire* object, effectively performing unsupervised segmentation.

---

## 3. DINOv2: Scaling and Unification

> **Motivation**: DINO v1 was great, but training on uncurated data (like raw web crawls) led to quality drops. Standard "foundation models" in NLP (GPT) work out-of-the-box; Vision needed the same.

### 3.1 The Three Pillars of DINOv2
1.  **Data Curation (LVD-142M)**
    *   They didn't just use more data; they built an automated pipeline to filter and balance a dataset of 142M images.
    *   Used image similarity to deduplicate and retrieve images similar to ImageNet-22k.

2.  **Architecture & Objective improvements**
    *   **iBOT Integration**: Added Masked Image Modeling (MIM) to DINO.
        *   DINO (Global): Matches global CLS token.
        *   iBOT (Local): Matches masked patch tokens.
    *   **Sinkhorn-Knopp Centering**: Replaced simple mean-centering with Sinkhorn-Knopp normalization (from SwAV) for better feature spread.
    *   **KoLeo Regularizer**: Maximizes the distance between features in a batch to ensure "uniform span" of the feature space.
    *   **High-Res Adaption**: A final short training stage at high resolution (e.g., 518x518) to sharpen localized features.

3.  **Efficient Implementation**
    *   FlashAttention, PyTorch 2.0 optimizations, and Fully Sharded Data Parallel (FSDP).

### 3.2 Key Results
DINOv2 provided a **"All-in-One" backbone**:
*   **Frozen features** outperform fine-tuned models on many tasks.
*   Works for Depth Estimation, Semantic Segmentation, and Instance Retrieval without any task-specific training.

---

## 4. DINOv3: The "Infinite Training" Paradox

> **The Problem with DINOv2**: When you train for *too long*, the global metrics (e.g., ImageNet classification) keep improving, BUT the **dense features** (patch-level understanding for segmentation/depth) start to degrade.

The authors of DINOv3 (2025) discovered that long training schedules cause "patch-level inconsistencies". The model becomes *too* abstract, losing the precise spatial coherence needed for dense tasks.

### 4.1 The Solution: Gram Anchoring
To fix this, DINOv3 introduces a regularization term called **Gram Anchoring**.

*   **Concept**: We want the features to evolve (to get better global performance), but we want the *relationship between patches* (the structure) to stay consistent with early, healthy training states.
*   **The "Gram Teacher"**: They take a snapshot of the teacher network from an earlier training step (where dense features were good).
*   **The Gram Matrix**: $G = X X^T$. Encodes the similarity between every pair of patches in an image.
*   **The Loss**:



    $$
    \mathcal{L}_{Gram} = || G_{student} - G_{GramTeacher} ||_F^2
    $$




This forces the student's current patch relationships to match the structural "fingerprint" of the Gram Teacher, even as the individual feature values change.

### 4.2 Algorithm: Gram Anchoring Strategy
1.  Train normally for first 11M iterations.
2.  **Snapshot**: Fix a "Gram Teacher" (provides the structural anchor).
3.  **Continue Training**: Add $\mathcal{L}_{Gram}$ to the loss.
4.  **Refinement**: Periodically update the Gram Teacher to "catch up" slightly, but always keeping it anchored to a stable dense-feature state.

### 4.3 Why "Gram"?
The Gram matrix captures the **style** or **texture** of feature interactions (famous from Style Transfer papers). Here, it captures the "geometric consistency" of the image representation.

### 4.4 Results
*   **No Compromise**: DINOv3 achieves SOTA on *both* global tasks (ImageNet) and dense tasks (Segmentation/Depth).
*   **Scalability**: Allows training "indefinitely" without the dense performance collapsing.

---

## 5. Visual Summary & Conclusion

### 5.1 Evolution Table

| Feature | DINO (v1) | DINOv2 | DINOv3 |
| :--- | :--- | :--- | :--- |
| **Year** | 2021 | 2023 | 2025 |
| **Core Method** | Self-Distillation | DINO + iBOT (MIM) | DINO + iBOT + **Gram Anchoring** |
| **Collapse Prevention** | Centering + Sharpening | Sinkhorn-Knopp + KoLeo | Same + Gram Regularization |
| **Data** | ImageNet (1M) | LVD-142M (Curated) | Scaled & Long-schedule |
| **Main Strength** | Unsupervised Segmentation | Robust Generalist Features | Infinite Training Stability |
| **Weakness** | Hard to scale to web data | Dense features degrade if trained too long | Complexity of multiple teachers |

### 5.2 Takeaway for Students
The DINO series represents the shift from **Contrastive** to **Generative/Distillative** learning in Vision.
*   **v1** taught us that simple rules (centering/sharpening) yield emergent intelligence.
*   **v2** taught us that **Data Quality** matters as much as the algorithm.
*   **v3** taught us that maximizing one metric (global loss) can hurt others (dense consistency), and we need **Explicit Structural Regularization** (Gram Anchoring) to balance them.

### 5.3 Further Reading
*   [DINOv1 Paper](https://arxiv.org/abs/2104.14294)
*   [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
*   [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
