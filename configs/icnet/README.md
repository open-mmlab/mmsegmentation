# ICNet

[ICNet for Real-time Semantic Segmentation on High-resolution Images](https://arxiv.org/abs/1704.08545)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/hszhao/ICNet">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/necks/ic_neck.py#L77">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

We focus on the challenging task of real-time semantic segmentation in this paper. It finds many practical applications and yet is with fundamental difficulty of reducing a large portion of computation for pixel-wise label inference. We propose an image cascade network (ICNet) that incorporates multi-resolution branches under proper label guidance to address this challenge. We provide in-depth analysis of our framework and introduce the cascade feature fusion unit to quickly achieve high-quality segmentation. Our system yields real-time inference on a single GPU card with decent quality results evaluated on challenging datasets like Cityscapes, CamVid and COCO-Stuff.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142901772-4570455d-7b27-44ae-a690-47dd9fde8445.png" width="70%"/>
</div>

## Citation

```bibtext
@inproceedings{zhao2018icnet,
  title={Icnet for real-time semantic segmentation on high-resolution images},
  author={Zhao, Hengshuang and Qi, Xiaojuan and Shen, Xiaoyong and Shi, Jianping and Jia, Jiaya},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={405--420},
  year={2018}
}
```

## Results and models

### Cityscapes

| Method           | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                             | download                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ICNet            | R-18-D8  | 832x832   |   80000 | 1.70     | 27.12          | 68.14 |         70.16 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r18-d8_832x832_80k_cityscapes.py)            | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r18-d8_832x832_80k_cityscapes/icnet_r18-d8_832x832_80k_cityscapes_20210925_225521-2e36638d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r18-d8_832x832_80k_cityscapes/icnet_r18-d8_832x832_80k_cityscapes_20210925_225521.log.json)                                             |
| ICNet            | R-18-D8  | 832x832   |  160000 | -        | -              | 71.64 |         74.18 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r18-d8_832x832_160k_cityscapes.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r18-d8_832x832_160k_cityscapes/icnet_r18-d8_832x832_160k_cityscapes_20210925_230153-2c6eb6e0.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r18-d8_832x832_160k_cityscapes/icnet_r18-d8_832x832_160k_cityscapes_20210925_230153.log.json)                                         |
| ICNet (in1k-pre) | R-18-D8  | 832x832   |   80000 | -        | -              | 72.51 |         74.78 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r18-d8_in1k-pre_832x832_80k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r18-d8_in1k-pre_832x832_80k_cityscapes/icnet_r18-d8_in1k-pre_832x832_80k_cityscapes_20210925_230354-1cbe3022.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r18-d8_in1k-pre_832x832_80k_cityscapes/icnet_r18-d8_in1k-pre_832x832_80k_cityscapes_20210925_230354.log.json)         |
| ICNet (in1k-pre) | R-18-D8  | 832x832   |  160000 | -        | -              | 74.43 |         76.72 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes_20210926_052702-619c8ae1.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes_20210926_052702.log.json)     |
| ICNet            | R-50-D8  | 832x832   |   80000 | 2.53     | 20.08          | 68.91 |         69.72 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r50-d8_832x832_80k_cityscapes.py)            | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r50-d8_832x832_80k_cityscapes/icnet_r50-d8_832x832_80k_cityscapes_20210926_044625-c6407341.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r50-d8_832x832_80k_cityscapes/icnet_r50-d8_832x832_80k_cityscapes_20210926_044625.log.json)                                             |
| ICNet            | R-50-D8  | 832x832   |  160000 | -        | -              | 73.82 |         75.67 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r50-d8_832x832_160k_cityscapes.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r50-d8_832x832_160k_cityscapes/icnet_r50-d8_832x832_160k_cityscapes_20210925_232612-a95f0d4e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r50-d8_832x832_160k_cityscapes/icnet_r50-d8_832x832_160k_cityscapes_20210925_232612.log.json)                                         |
| ICNet (in1k-pre) | R-50-D8  | 832x832   |   80000 | -        | -              | 74.58 |         76.41 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r50-d8_in1k-pre_832x832_80k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r50-d8_in1k-pre_832x832_80k_cityscapes/icnet_r50-d8_in1k-pre_832x832_80k_cityscapes_20210926_032943-1743dc7b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r50-d8_in1k-pre_832x832_80k_cityscapes/icnet_r50-d8_in1k-pre_832x832_80k_cityscapes_20210926_032943.log.json)         |
| ICNet (in1k-pre) | R-50-D8  | 832x832   |  160000 | -        | -              | 76.29 |         78.09 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes_20210926_042715-ce310aea.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes_20210926_042715.log.json)     |
| ICNet            | R-101-D8 | 832x832   |   80000 | 3.08     | 16.95          | 70.28 |         71.95 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r101-d8_832x832_80k_cityscapes.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r101-d8_832x832_80k_cityscapes/icnet_r101-d8_832x832_80k_cityscapes_20210926_072447-b52f936e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r101-d8_832x832_80k_cityscapes/icnet_r101-d8_832x832_80k_cityscapes_20210926_072447.log.json)                                         |
| ICNet            | R-101-D8 | 832x832   |  160000 | -        | -              | 73.80 |         76.10 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r101-d8_832x832_160k_cityscapes.py)          | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r101-d8_832x832_160k_cityscapes/icnet_r101-d8_832x832_160k_cityscapes_20210926_092350-3a1ebf1a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r101-d8_832x832_160k_cityscapes/icnet_r101-d8_832x832_160k_cityscapes_20210926_092350.log.json)                                     |
| ICNet (in1k-pre) | R-101-D8 | 832x832   |   80000 | -        | -              | 75.57 |         77.86 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r101-d8_in1k-pre_832x832_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r101-d8_in1k-pre_832x832_80k_cityscapes/icnet_r101-d8_in1k-pre_832x832_80k_cityscapes_20210926_020414-7ceb12c5.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r101-d8_in1k-pre_832x832_80k_cityscapes/icnet_r101-d8_in1k-pre_832x832_80k_cityscapes_20210926_020414.log.json)     |
| ICNet (in1k-pre) | R-101-D8 | 832x832   |  160000 | -        | -              | 76.15 |         77.98 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/icnet/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes_20210925_232612-9484ae8a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/icnet/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes_20210925_232612.log.json) |

Note: `in1k-pre` means pretrained model is used.
