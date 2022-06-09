# ResNeSt

[ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/zhanghang1989/ResNeSt">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/resnest.py#L271">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

It is well known that featuremap attention and multi-path representation are important for visual recognition. In this paper, we present a modularized architecture, which applies the channel-wise attention on different network branches to leverage their success in capturing cross-feature interactions and learning diverse representations. Our design results in a simple and unified computation block, which can be parameterized using only a few variables. Our model, named ResNeSt, outperforms EfficientNet in accuracy and latency trade-off on image classification. In addition, ResNeSt has achieved superior transfer learning results on several public benchmarks serving as the backbone, and has been adopted by the winning entries of COCO-LVIS challenge. The source code for complete system and pretrained models are publicly available.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142902526-3cf33345-7e40-47a6-985e-4381857e21df.png" width="60%"/>
</div>

## Citation

```bibtex
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```

## Results and models

### Cityscapes

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                              | download                                                                                                                                                                                                                                                                                                                                                                               |
| ---------- | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN        | S-101-D8 | 512x1024  |   80000 |     11.4 | 2.39           | 77.56 | 78.98         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest/fcn_s101-d8_512x1024_80k_cityscapes.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x1024_80k_cityscapes/fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x1024_80k_cityscapes/fcn_s101-d8_512x1024_80k_cityscapes-20200807_140631.log.json)                                         |
| PSPNet     | S-101-D8 | 512x1024  |   80000 |     11.8 | 2.52           | 78.57 | 79.19         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x1024_80k_cityscapes/pspnet_s101-d8_512x1024_80k_cityscapes_20200807_140631-c75f3b99.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x1024_80k_cityscapes/pspnet_s101-d8_512x1024_80k_cityscapes-20200807_140631.log.json)                             |
| DeepLabV3  | S-101-D8 | 512x1024  |   80000 |     11.9 | 1.88           | 79.67 | 80.51         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes/deeplabv3_s101-d8_512x1024_80k_cityscapes_20200807_144429-b73c4270.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x1024_80k_cityscapes/deeplabv3_s101-d8_512x1024_80k_cityscapes-20200807_144429.log.json)                 |
| DeepLabV3+ | S-101-D8 | 512x1024  |   80000 |     13.2 | 2.36           | 79.62 | 80.27         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest/deeplabv3plus_s101-d8_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3plus_s101-d8_512x1024_80k_cityscapes/deeplabv3plus_s101-d8_512x1024_80k_cityscapes_20200807_144429-1239eb43.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3plus_s101-d8_512x1024_80k_cityscapes/deeplabv3plus_s101-d8_512x1024_80k_cityscapes-20200807_144429.log.json) |

### ADE20K

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                               |
| ---------- | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN        | S-101-D8 | 512x512   |  160000 |     14.2 | 12.86          | 45.62 | 46.16         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest/fcn_s101-d8_512x512_160k_ade20k.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x512_160k_ade20k/fcn_s101-d8_512x512_160k_ade20k_20200807_145416-d3160329.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/fcn_s101-d8_512x512_160k_ade20k/fcn_s101-d8_512x512_160k_ade20k-20200807_145416.log.json)                                         |
| PSPNet     | S-101-D8 | 512x512   |  160000 |     14.2 | 13.02          | 45.44 | 46.28         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x512_160k_ade20k/pspnet_s101-d8_512x512_160k_ade20k_20200807_145416-a6daa92a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/pspnet_s101-d8_512x512_160k_ade20k/pspnet_s101-d8_512x512_160k_ade20k-20200807_145416.log.json)                             |
| DeepLabV3  | S-101-D8 | 512x512   |  160000 |     14.6 | 9.28           | 45.71 | 46.59         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest/deeplabv3_s101-d8_512x512_160k_ade20k.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x512_160k_ade20k/deeplabv3_s101-d8_512x512_160k_ade20k_20200807_144503-17ecabe5.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3_s101-d8_512x512_160k_ade20k/deeplabv3_s101-d8_512x512_160k_ade20k-20200807_144503.log.json)                 |
| DeepLabV3+ | S-101-D8 | 512x512   |  160000 |     16.2 | 11.96          | 46.47 | 47.27         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest/deeplabv3plus_s101-d8_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3plus_s101-d8_512x512_160k_ade20k/deeplabv3plus_s101-d8_512x512_160k_ade20k_20200807_144503-27b26226.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/resnest/deeplabv3plus_s101-d8_512x512_160k_ade20k/deeplabv3plus_s101-d8_512x512_160k_ade20k-20200807_144503.log.json) |
