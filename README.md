# As-ViT: Auto-scaling Vision Transformers without Training [[PDF](https://openreview.net/pdf?id=H94a1_Pyr-6)]

<!-- [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/VITA-Group/TENAS.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/VITA-Group/TENAS/context:python) -->
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Wuyang Chen, Wei Huang, Xianzhi Du, Xiaodan Song, Zhangyang Wang, Denny Zhou

In ICLR 2022.


**Note**: We implemented topology search (sec. 3.3) and scaling (sec. 3.4) in this code base in PyTorch. Our training code is based on Tensorflow and Keras on TPU, which will be released soon.



## Overview

We present As-ViT, a framework that unifies the automatic architecture design and scaling for ViT (vision transformer), in a training-free strategy.

Highlights:
* **Trainig-free ViT Architecture Design**: we design a "seed" ViT topology by leveraging a training-free search process. This extremely fast search is fulfilled by our comprehensive study of ViT's network complexity (length distorsion), yielding a strong Kendall-tau correlation with ground-truth accuracies.
* **Trainig-free ViT Architecture Scaling**: starting from the "seed" topology, we automate the scaling rule for ViTs by growing widths/depths to different ViT layers. This will generate a series of architectures with different numbers of parameters in a single run.
* **Efficient ViT Training via Progressive Tokenization**: we observe that ViTs can tolerate coarse tokenization in early training stages,  and further propose to train ViTs faster and cheaper with a progressive tokenization strategy.
<!-- * **SOTA**: TE-NAS achieved extremely fast search speed (one 1080Ti, 20 minutes on NAS-Bench-201 space / four hours on DARTS space on ImageNet) and maintains competitive accuracy. -->

<!--
<p align="center">
<img src="images/????.png" alt="201" width="550"/></br>
</p>
<p align="center">
<img src="images/????.png" alt="darts_cifar10" width="550"/></br>
</p>
<p align="center">
<img src="images/????.png" alt="darts_imagenet" width="550"/></br>
</p>
-->

<!--
## Methods

<p align="center">
<img src="images/????.png" alt="algorithm" width="800"/></br>
</p>
-->


## Prerequisites
- Ubuntu 18.04
- Python 3.6.9
- CUDA 11.0 (lower versions may work but were not tested)
- NVIDIA GPU + CuDNN v7.6

This repository has been tested on V100 GPU. Configurations may need to be changed on different platforms.


## Installation
* Clone this repo:
```bash
git clone https://github.com/VITA-Grou/AsViT.git
cd TEGNAS
```
* Install dependencies:
```bash
pip install -r requirements.txt
```


## 1. Seed As-ViT Topology Search
```bash
CUDA_VISIBLE_DEVICES=0 python ./search/reinforce.py --save_dir ./output/REINFORCE-imagenet --data_path /path/to/imagenet
```
This job will return you a seed topology. For example, our search seed topology is `8,2,3|4,1,2|4,1,4|4,1,6|32`, which can be explained as below:

<table><thead><tr><th colspan="3">Stage1</th><th colspan="3">Stage2</th><th colspan="3">Stage3</th><th colspan="3">Stage4</th><th rowspan="2">Head</th></tr><tr><th>Kernel K1</th><th>Split S1</th><th>Expansion E1</th><th>Kernel K2</th><th>Split S2</th><th>Expansion E2</th><th>Kernel K3</th><th>Split S3</th><th>Expansion E3</th><th>Kernel K4</th><th>Split S4</th><th>Expansion E4</th></tr></thead><tbody><tr><td>8</td><td>2</td><td>3</td><td>4</td><td>1</td><td>2</td><td>4</td><td>1</td><td>4</td><td>4</td><td>1</td><td>6</td><td>32</td></tr></tbody></table>

## 2. Scaling
```bash
CUDA_VISIBLE_DEVICES=0 python ./search/grow.py --save_dir ./output/GROW-imagenet \
--arch "[arch]" --data_path /path/to/imagenet
```
Here `[arch]` is the seed topology (output from step 1 above).
This job will return you a series of topologies. For example, our largest topology (As-ViT Large) is `8,2,3,5|4,1,2,2|4,1,4,5|4,1,6,2|32,180`, which can be explained as below:

<table><thead><tr><th colspan="4">Stage1</th><th colspan="4">Stage2</th><th colspan="4">Stage3</th><th colspan="4">Stage4</th><th rowspan="2">Head</th><th rowspan="2">Initial Hidden Size</th></tr><tr><th>Kernel K1</th><th>Split S1</th><th>Expansion E1</th><th>Layers L1</th><th>Kernel K2</th><th>Split S2</th><th>Expansion E2</th><th>Layers L2</th><th>Kernel K3</th><th>Split S3</th><th>Expansion E3</th><th>Layers L3</th><th>Kernel K4</th><th>Split S4</th><th>Expansion E4</th><th>Layers L4</th></tr></thead><tbody><tr><td>8</td><td>2</td><td>3</td><td>5</td><td>4</td><td>1</td><td>2</td><td>2</td><td>4</td><td>1</td><td>4</td><td>5</td><td>4</td><td>1</td><td>6</td><td>2</td><td>32</td><td>180</td></tr></tbody></table>


### 3. Evaluation
Tensorflow and Keras code for training on TPU. To be released soon.


## Citation
```
@inproceedings{chen2021asvit,
  title={Auto-scaling Vision Transformers without Training},
  author={Chen, Wuyang and Huang, Wei and Du, Xianzhi and Song, Xiaodan and Wang, Zhangyang and Zhou, Denny},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
