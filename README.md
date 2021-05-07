This repository contains the implementation of:
Actional-Structural Graph Convolutional Networks for Skeleton-based Action Recognition. [Paper](https://arxiv.org/pdf/1904.12659.pdf)

![image](https://github.com/limaosen0/AS-GCN/blob/master/img/pipeline.png)

Abstract: Action recognition with skeleton data has recently attracted much attention in computer vision. Previous studies are mostly based on fixed skeleton graphs, only capturing local physical dependencies among joints, which may miss implicit joint correlations. To capture richer dependencies, we introduce an encoder-decoder structure, called A-link inference module, to capture action-specific latent dependencies, i.e. actional links, directly from actions. We also extend the existing skeleton graphs to represent higherorder dependencies, i.e. structural links. Combing the two types of links into a generalized skeleton graph, we further propose the actional-structural graph convolution network (AS-GCN), which stacks actional-structural graph convolution and temporal convolution as a basic building block, to learn both spatial and temporal features for action recognition. A future pose prediction head is added in parallel to the recognition head to help capture more detailed action patterns through self-supervision. We validate AS-GCN in action recognition using two skeleton data sets, NTU-RGB+D and Kinetics. The proposed AS-GCN achieves consistently large improvement compared to the state-of-the-art methods. As a side product, AS-GCN also shows promising results for future pose prediction.

In this repo, we show the example of model on NTU-RGB+D dataset.

# Experiment Requirement
* Python 3.6
* Pytorch 0.4.1
* pyyaml
* argparse
* numpy

# Environments
We use the similar input/output interface and system configuration like ST-GCN, where the torchlight module should be set up.

Run
```
cd torchlight, python setup.py, cd ..
```


# Data Preparing
For NTU-RGB+D dataset, you can download it from [NTU-RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). And put the dataset in the file path:
```
'./data/NTU-RGB+D/nturgb+d_skeletons/'
```
Then, run the preprocessing program to generate the input data, which is very important.
```
python ./data_gen/ntu_gen_preprocess.py
```

# Training and Testing
With this repo, you can pretrain AIM and save the module at first; then run the code to train the main pipleline of AS-GCN. For the recommended benchmark of Cross-Subject in NTU-RGB+D,
```
PretrainAIM: python main.py recognition -c config/as_gcn/ntu-xsub/train_aim.yaml
TrainMainPipeline: python main.py recognition -c config/as_gcn/ntu-xsub/train.yaml
Test: python main.py recognition -c config/as_gcn/ntu-xsub/test.yaml
```

For Cross-View,
```
PretrainAIM: python main.py recognition -c config/as_gcn/ntu-xview/train_aim.yaml
TrainMainPipeline: python main.py recognition -c config/as_gcn/ntu-xview/train.yaml
Test: python main.py recognition -c config/as_gcn/ntu-xview/test.yaml
```

# Acknowledgement
Thanks for the framework provided by 'yysijie/st-gcn', which is source code of the published work [ST-GCN](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17135) in AAAI-2018. The github repo is [ST-GCN code](https://github.com/yysijie/st-gcn). We borrow the framework and interface from the code.

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
