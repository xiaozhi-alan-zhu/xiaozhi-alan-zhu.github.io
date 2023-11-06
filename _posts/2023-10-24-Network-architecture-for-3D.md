---
layout: distill
title: Neural Network Architecture for 3D Data Feature Extraction
date: 2023-10-23
description: PVCNN, PointNet, PointNet++, 3D-UNet
tag: ML
bibliography: reference.bib
toc:
    - name: Introduction
    - name: PointNet
    - name: PointNet++
    - name: PVCNN
---

# Introduction
3D data, unlike 2D image, is usually more complicated and requires more computational resource(for similar tasks).
More importantly, there are multiple [different representation methods](https://xiaozhi-alan-zhu.github.io/blog/2023/Neural-representation-shape/) for 3D shapes and most popular ones inlcude multi-view, volumetric, point cloud, mesh and etc.
Different 3D asset generation method gives different data form: multi-view data is generated from multi-camera data, point cloud is typically captured by LiDAR sensors, mesh data is typically building block for software such as Blender, Autodesk and etc.

# PointNet
A point cloud is represented as a set of 3D points $$\{p_i\mid i=1, \cdots, n\}$$, where each point $$p_i$$ is a vector of its $$(x,y,z)$$ coordinate plus extra feature channels such as color, normal etc.
Point cloud data has three main properties:
1. Unordered: image pixels are ordered but points in point cloud is not.
2. Interaction among points: points are not isolated and local structures should be captured from nearby points.
3. Invariance under transformations: data's certain properties(catagories, segmentation and etc) are not changed over certain transformation(rotation, translation).

PointNet model, proposed in <d-cite key="qi2017pointnet"></d-cite>, is proposed to resolve these gaps.

## Model architecture
The paper introduced specific structure(induced bias) to deal with the challenges mentioned previously.
The overall model structure is presented in figure 1.
<figure>
  <img src="../../../assets/img/3d-learning/pointnet-flowchart.png" style="width:100%">
  <figcaption>Fig.1 - Flowchart of PointNet. It has two separate paths for classification and segmentation tasks respectively. </figcaption>
</figure>

**Symmetry function for unordered input**:

A symmetric function is a simple solution for unordered point cloud.
By symmetric, we mean that $$f(x_1, x_2, \cdots, x_n)=f(x_{i_1}, \cdots, x_{i_n})$$ where $$\{i_1, \cdots, i_n\}$$ is a random permutation of $$\{1, 2, \cdots, n\}$$.
In the context of this paper, the symmetric function applied to the point cloud takes the following form

$$
\begin{equation}
f(\{x_1, \cdots, x_n\})=g(h(x_1),\cdots, h(x_n)),
\end{equation}
$$

where $$f:2^{\mathbb{R}^N}\rightarrow\mathbb{R},~h:\mathbb{R}^N\rightarrow\mathbb{R}^K$$ and $$g:\mathbb{R}^K\times\cdots\times\mathbb{R}^K$$.

**Local and global information aggregation**:

Local and global information both contained important signals about point cloud.
However, different tasks can have different dependency on these signals:
1. For classification, intuitively, only global features are needed.
But for tasks like segmentation, local features is as important as global feature.
The choice of the PointNet is to concatenate intermediate pointwise layer output and global feature.
2. For segmentation, both local and global signal can be useful.
The paper's solution is to concatenate the global feature with pointwise intermediate hidden layer outputs.

**Joint alignment network**:

Ideally, prediction over a point cloud should be invariant with regard to the orientation of the point cloud.
To be more specific, the prediction result(i.e. label for classification task) should be the same if we rotate or translate the point cloud.
This requires some spatial transform in the model.
The paper introduced mini-network(called T-net) to predict an affine transformation matrix and directly apply this transformation to the coordinates of input points.
To contrain the predicted matrix is a spatial transformation matrix, an additional regularization term is added:

$$
L_\mathrm{reg}=\|I-AA^T\|_F^2
$$

## Theoretic properties
Two interesting theoretical properties is proved in the paper:
1. **Universal approximation**: 
The paper showed that the composition of a continuous function $$h$$ and a symmetric function $$g$$ still enjoy the universal approximation property(i.e. it can approximate any contiuous function w.r.t. Hausdorff distance).

2. **Bottleneck dimension and stability**: 
The paper showed that there exist a critical subset of each point cloud, of which the number of points is less than the dimension of the max pooling layer K, and noise added to points outside of this critical subset is not likely to change the final prediction.
This property suggest that the PointNet model is likely to learn to summarize a shape by a sparse set of key points.

# PointNet++
An obvious drawback of PointNet is that the local structual information is not captured by the model.
Certainly, the pointwise embedding in the intermediate layer embedded some local information.
But the important structural signals(such as local curvature, normal vector and etc) is not captured.
PointNet++, proposed in <d-cite key="qi2017pointnet++"></d-cite>, aims to increase model's ability to recognize fine-grained patterns and generalizability to complex scenes.

## Formulation
The PointNet++ model introduced hierechical feature learning process to extract both local and global information.
It incorporated different down-sampling mechanism to deal with non-uniform point distribution on the shape.
The paper also discussed, for segmentation task, how to propagate global information(signal in higher abstract level) to pointwise embedding.

<figure>
  <img src="../../../assets/img/3d-learning/pointnet++-flowchart.png" style="width:100%">
  <figcaption>Fig.2 - Flowchart of PointNet. It also has two separate paths for classification and segmentation tasks respectively. </figcaption>
</figure>

**Hierarchical Point Set Feature Learning**:

There are three components of the hierarchical point set feature learning process:
1. Sampling layer: 
Farthest point sampling(FPS) is used to downsample point set.
The subset $$\{x_{i_1},\cdots, x_{i_m}\}$$ is subsampled from $$\{x_1, x_2,\cdots, x_n\}$$ such that $$x_{i_j}$$ is the most distant point from the set $$\{x_{i_1},\cdots, x_{i_{j-1}}\}$$.
An implementation can be found [here](http://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.farthest_point_down_sample).
An important different from downsampling in CNN is that the receptive fields are depended on data for FPS.
2. Grouping layer:
We refer to the points sampled from previous sampling layer as "centroids".
At this layer, the model group the features of points surround each centroids.
Each centroid can have different receptive fields and each point can be grouped into multiple different centroids.
The input are point cloud feature matrix of size $$N\times (d+C)$$ and centroids location matrix $$N'\times (d+C)$$.
The output are tensor of size $$N'\times K \times(d+C)$$.
The K is the number of points surround a centroid and is different for different centroids.
3. PointNet layer:
At this layer, for each centroids, the input matrix $$ K \times (d + C)$$ is mapped to a vector of size $$C'$$.

**Robust Feature Learning under Non-Uniform Sampling Dnesity**

The sampling layer has picked a subset of points from the point cloud as centroids and each of them has their own receptive fields(with intersection).
The grouping layer need to resolve the problem of how to aggregate information/features in a receptive field.
The paper proposed two grouping strategies:
1. **Multi-scale grouping** (MSG): 
The MSG method apply grouping layers with different scales to extract features of each scale.
Features at different scales are concatenated to form a multi-scale feature.
It also introduced a **random input dropout** process that dropping out input points with a probability $$\theta$$ sampled from $$U[0, p]$$.
This process can create training sets of various sparsity and varying uniformity(randomness in dropout).
The authors' implementation can be found [here](https://github.com/charlesq34/pointnet2/blob/master/utils/pointnet_util.py#L156).
2. **Multi-resolution grouping** (MRG):
In this grouping method, the features of a region at some level $$L_i$$ is a concatenation of tow vectors.
One vector (left in figure) is obtained by summarizing the features at each subregion from the lower level $$L_{i−1}$$ using the set abstraction level.
The other vector (right) is the feature that is obtained by directly processing all raw points in the local region using a single PointNet.

<figure>
  <img src="../../../assets/img/3d-learning/pointnet++-msg-mrg-demo.png" style="width:50%" class="center">
  <figcaption>Fig.3 - MSG vs MRG, image source <d-cite key="qi2017pointnet++"></d-cite>. </figcaption>
</figure>

**Point Feature Propagation for Set Segmentation**

For the segmentation task, the model needs to propagate features from subsampled points to the original points.
The feature propagation is achieved by interpolating feature values $$f$$ of $$N_l$$ points at coordinates of $$N_{l-1}$$ points.
Each channel(or dimension) in the feature is computed as follows,
$$
\begin{equation}
f^{(j)}(x)=\frac{\sum_{i=1}^k w_i(x) f_i^{(j)}}{\sum_{i=1}^k w_i(x)} \quad \text { where } \quad w_i(x)=\frac{1}{d\left(x, x_i\right)^p}, j=1, \ldots, C.
\end{equation}
$$


# Point-Voxel CNN(PVCNN)

Previous methods are focused on discussing extraction of features from point cloud.
The PVCNN model, proposed in <d-cite key="liu2019point"></d-cite>, emphasized its performance on efficiency.
In this follow graph, the latency and memory usage have been compared between PVCNN and other popular choices.
<figure>
  <img src="../../../assets/img/3d-learning/pvcnn-efficiency.png" style="width:100%" class="center">
  <figcaption>Fig.4 - Efficiency comparison between different model architecture, image source <d-cite key="liu2019point"></d-cite>. </figcaption>
</figure>

**Why efficiency is a problem?**: The paper summarized the common reasons that cause model slow and take lots of memory for 3D features.
1. High resolution requires high memory cost for voxel-based method.
The low resolution can lead to information loss and points in the same voxel grid are not distinguishable.
The paper estimated that a single GPU(with 12GB memory) can only afford voxel of 64 in each dimension and lead to 42% of information loss.

2. Irregular memory access and dynamic kernel overhead for point-based models.
Models like PointNet++ utilizes neighborhood information in the point domain which lead to the irregular memory access pattern and becomes the fficiency bottlenecks.
Furthemore, the computation of kernel $$\mathcal{K}(x_o, x_j)$$ is more costful, since, unlike 3D volumetric convolutions where neighborhood index is readily accessible, the model need to find neighbors on the fly.

## Formulation

The PVCNN architecture introduced in this paper combined advantages of both point-based and voxel-based methods by disentangles the fine-grained feature transformation and the coarse-grained neighbor aggregation.

### Voxel-based fature aggregation:
The voxel-based feature aggregation process is demonstrated in the upper half of Fig 5.
It consist following sub-routines:
1. Normalization: The original point clouds $$\{\mathbb{p}_k\}$$ can have different scale, center location and etc.
In this step, the model translate all points in a point cloud with the gravity center as origin.
The scale the points by dividing each coordinates by $$\max_i ||\mathbb{p}_i||^2$$.
Finally, we denote the output normalized points as $$\{\hat{\mathbb{p}}_k)\}$$.
2. Voxelization: 
Pointwise convolution is costful since finding neighbor is not trivial for data represented in the form of point cloud.
In this step, points are grouped into voxel grids $$\{V_{u,v,w}\}$$ and their features are aggregated by taking average.
Let $$f_k$$ denotes point $$\hat{p}_k=(\hat{x}_k, \hat{y}_k, \hat{z}_k)$$'s features, then the feature of voxel grid $$(u, v, w)$$ is
$$
\begin{equation}
\boldsymbol{V}_{u, v, w, c}=\frac{1}{N_{u, v, w}} \sum_{k=1}^n \mathbb{I}\left[\text { floor }\left(\hat{\boldsymbol{x}}_k \times r\right)=u, \text { floor }\left(\hat{\boldsymbol{y}}_k \times r\right)=v, \text { floor }\left(\hat{\boldsymbol{z}}_k \times r\right)=w\right] \times \boldsymbol{f}_{k, c}.
\end{equation}
$$
3. Feature aggregation:
Conventional 3D convolution can now be applied to voxel grid $$\{V_{u,v,w}\}$$ at much lower cost.
4. Devoxelization: 
In this step, the model map voxel feature to point feature trhough trilinear interpolation to ensure that the features mapped to each point are distinct.

<figure>
  <img src="../../../assets/img/3d-learning/pvcnn-flowchart.png" style="width:100%" class="center">
  <figcaption>Fig.5 - Flowchart of PVCNN, image source <d-cite key="liu2019point"></d-cite>. </figcaption>
</figure>

### Point-based feature transformation
This is represented in the bottom half of the Fig. 5.
The process is very straight forward and is directly apply MLP to point-wise features.

### Feature fusion
In this step, the model fuse both individual point features and aggregated neighborhood information with addition.
