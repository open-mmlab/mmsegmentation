# CGNet

[CGNet: A Light-weight Context Guided Network for Semantic Segmentation](https://arxiv.org/abs/1811.08201)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/wutianyiRosun/CGNet">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/cgnet.py#L187">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The demand of applying semantic segmentation model on mobile devices has been increasing rapidly. Current state-of-the-art networks have enormous amount of parameters hence unsuitable for mobile devices, while other small memory footprint models follow the spirit of classification network and ignore the inherent characteristic of semantic segmentation. To tackle this problem, we propose a novel Context Guided Network (CGNet), which is a light-weight and efficient network for semantic segmentation. We first propose the Context Guided (CG) block, which learns the joint feature of both local feature and surrounding context, and further improves the joint feature with the global context. Based on the CG block, we develop CGNet which captures contextual information in all stages of the network and is specially tailored for increasing segmentation accuracy. CGNet is also elaborately designed to reduce the number of parameters and save memory footprint. Under an equivalent number of parameters, the proposed CGNet significantly outperforms existing segmentation networks. Extensive experiments on Cityscapes and CamVid datasets verify the effectiveness of the proposed approach. Specifically, without any post-processing and multi-scale testing, the proposed CGNet achieves 64.8% mean IoU on Cityscapes with less than 0.5 M parameters. The source code for the complete system can be found at [this https URL](https://github.com/wutianyiRosun/CGNet).

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142900351-89559574-79cc-4f57-8f69-5d88765ec38d.png" width="80%"/>
</div>

## Citation

```bibtext
@article{wu2020cgnet,
  title={Cgnet: A light-weight context guided network for semantic segmentation},
  author={Wu, Tianyi and Tang, Sheng and Zhang, Rui and Cao, Juan and Zhang, Yongdong},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={1169--1179},
  year={2020},
  publisher={IEEE}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                            | download                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CGNet  | M3N21    | 680x680   |   60000 | 7.5      | 30.51          | 65.63 |         68.04 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/cgnet/cgnet_680x680_60k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/cgnet/cgnet_680x680_60k_cityscapes/cgnet_680x680_60k_cityscapes_20201101_110253-4c0b2f2d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/cgnet/cgnet_680x680_60k_cityscapes/cgnet_680x680_60k_cityscapes-20201101_110253.log.json)     |
| CGNet  | M3N21    | 512x1024  |   60000 | 8.3      | 31.14          | 68.27 |         70.33 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/cgnet/cgnet_512x1024_60k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/cgnet/cgnet_512x1024_60k_cityscapes/cgnet_512x1024_60k_cityscapes_20201101_110254-124ea03b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/cgnet/cgnet_512x1024_60k_cityscapes/cgnet_512x1024_60k_cityscapes-20201101_110254.log.json) |
