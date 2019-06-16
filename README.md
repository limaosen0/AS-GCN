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

# Training and Testing
```
Train: python main.py recognition -c config/as_gcn/ntu-xsub/train.yaml
Test: python main.py recognition -c config/as_gcn/ntu-xsub/test.yaml
```
# Data
For NTU-RGB+D dataset, you can download it from [NTU-RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

# Environments
We use the similar input/output interface and system configuration like ST-GCN, where the torchlight module should be set up.


# Citation
If you use this code, please cite our paper:
```
@InProceedings{Li_2019_CVPR,
author = {Li, Maosen and Chen, Siheng and Chen, Xu and Zhang, Ya and Wang, Yanfeng and Tian, Qi},
title = {Actional-Structural Graph Convolutional Networks for Skeleton-Based Action Recognition},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
