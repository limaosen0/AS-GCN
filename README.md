This reposity contains the implementation code of:
Actional-Structural Graph Convolutional Networks for Skeleton-based Action Recognition. [Paper](https://arxiv.org/pdf/1904.12659.pdf)

![image](https://github.com/limaosen0/AS-GCN/blob/master/img/pipeline.png)

Abstract: Action recognition with skeleton data has recently attracted much attention in computer vision. Previous studies are mostly based on fixed skeleton graphs, only capturing local physical dependencies among joints, which may miss implicit joint correlations. To capture richer dependencies, we introduce an encoder-decoder structure, called A-link inference module, to capture action-specific latent dependencies, i.e. actional links, directly from actions. We also extend the existing skeleton graphs to represent higherorder dependencies, i.e. structural links. Combing the two types of links into a generalized skeleton graph, we further propose the actional-structural graph convolution network (AS-GCN), which stacks actional-structural graph convolution and temporal convolution as a basic building block, to learn both spatial and temporal features for action recognition. A future pose prediction head is added in parallel to the recognition head to help capture more detailed action patterns through self-supervision. We validate AS-GCN in action recognition using two skeleton data sets, NTURGB+D and Kinetics. The proposed AS-GCN achieves consistently large improvement compared to the state-of-the-art methods. As a side product, AS-GCN also shows promising results for future pose prediction.

# Experiment Requirement
* Python 3.6
* Pytorch 0.4.1
* pyyaml
* argparse
* numpy


# Citation
If you use this code, please cite our paper:
 
```
@ARTICLE{sss,
author={Ya\u{g}{\i}z Aksoy and Tae-Hyun Oh and Sylvain Paris and Marc Pollefeys and Wojciech Matusik},
title={Semantic Soft Segmentation},
journal={ACM Transactions on Graphics (Proc. SIGGRAPH)},
year={2018},
pages = {72:1-72:13},
volume = {37},
number = {4}
}
```
