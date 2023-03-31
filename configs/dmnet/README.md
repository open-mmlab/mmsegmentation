# DMNet

> [Dynamic Multi-scale Filters for Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/Junjun2016/DMNet">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/dm_head.py#L93">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Multi-scale representation provides an effective way toaddress scale variation of objects and stuff in semantic seg-mentation. Previous works construct multi-scale represen-tation by utilizing different filter sizes, expanding filter sizeswith dilated filters or pooling grids, and the parameters ofthese filters are fixed after training. These methods oftensuffer from heavy computational cost or have more param-eters, and are not adaptive to the input image during in-ference. To address these problems, this paper proposes aDynamic Multi-scale Network (DMNet) to adaptively cap-ture multi-scale contents for predicting pixel-level semanticlabels. DMNet is composed of multiple Dynamic Convolu-tional Modules (DCMs) arranged in parallel, each of whichexploits context-aware filters to estimate semantic represen-tation for a specific scale. The outputs of multiple DCMsare further integrated for final segmentation. We conductextensive experiments to evaluate our DMNet on three chal-lenging semantic segmentation and scene parsing datasets,PASCAL VOC 2012, Pascal-Context, and ADE20K. DMNetachieves a new record 84.4% mIoU on PASCAL VOC 2012test set without MS COCO pre-trained and post-processing,and also obtains state-of-the-art performance on Pascal-Context and ADE20K.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142900781-6215763f-8b71-4e0b-a6b1-c41372db2aa0.png" width="70%"/>
</div>

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                       | download                                                                                                                                                                                                                                                                                                                                           |
| ------ | -------- | --------- | ------: | -------- | -------------- | ------ | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DMNet  | R-50-D8  | 512x1024  |   40000 | 7.0      | 3.66           | V100   | 77.78 |         79.14 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r50-d8_4xb2-40k_cityscapes-512x1024.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_40k_cityscapes/dmnet_r50-d8_512x1024_40k_cityscapes_20201215_042326-615373cf.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_40k_cityscapes/dmnet_r50-d8_512x1024_40k_cityscapes-20201215_042326.log.json)     |
| DMNet  | R-101-D8 | 512x1024  |   40000 | 10.6     | 2.54           | V100   | 78.37 |         79.72 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r101-d8_4xb2-40k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x1024_40k_cityscapes/dmnet_r101-d8_512x1024_40k_cityscapes_20201215_043100-8291e976.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x1024_40k_cityscapes/dmnet_r101-d8_512x1024_40k_cityscapes-20201215_043100.log.json) |
| DMNet  | R-50-D8  | 769x769   |   40000 | 7.9      | 1.57           | V100   | 78.49 |         80.27 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r50-d8_4xb2-40k_cityscapes-769x769.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_769x769_40k_cityscapes/dmnet_r50-d8_769x769_40k_cityscapes_20201215_093706-e7f0e23e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_769x769_40k_cityscapes/dmnet_r50-d8_769x769_40k_cityscapes-20201215_093706.log.json)         |
| DMNet  | R-101-D8 | 769x769   |   40000 | 12.0     | 1.01           | V100   | 77.62 |         78.94 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r101-d8_4xb2-40k_cityscapes-769x769.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_769x769_40k_cityscapes/dmnet_r101-d8_769x769_40k_cityscapes_20201215_081348-a74261f6.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_769x769_40k_cityscapes/dmnet_r101-d8_769x769_40k_cityscapes-20201215_081348.log.json)     |
| DMNet  | R-50-D8  | 512x1024  |   80000 | -        | -              | V100   | 79.07 |         80.22 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r50-d8_4xb2-80k_cityscapes-512x1024.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_80k_cityscapes/dmnet_r50-d8_512x1024_80k_cityscapes_20201215_053728-3c8893b9.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_80k_cityscapes/dmnet_r50-d8_512x1024_80k_cityscapes-20201215_053728.log.json)     |
| DMNet  | R-101-D8 | 512x1024  |   80000 | -        | -              | V100   | 79.64 |         80.67 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r101-d8_4xb2-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x1024_80k_cityscapes/dmnet_r101-d8_512x1024_80k_cityscapes_20201215_031718-fa081cb8.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x1024_80k_cityscapes/dmnet_r101-d8_512x1024_80k_cityscapes-20201215_031718.log.json) |
| DMNet  | R-50-D8  | 769x769   |   80000 | -        | -              | V100   | 79.22 |         80.55 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r50-d8_4xb2-80k_cityscapes-769x769.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_769x769_80k_cityscapes/dmnet_r50-d8_769x769_80k_cityscapes_20201215_034006-6060840e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_769x769_80k_cityscapes/dmnet_r50-d8_769x769_80k_cityscapes-20201215_034006.log.json)         |
| DMNet  | R-101-D8 | 769x769   |   80000 | -        | -              | V100   | 79.19 |         80.65 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r101-d8_4xb2-80k_cityscapes-769x769.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_769x769_80k_cityscapes/dmnet_r101-d8_769x769_80k_cityscapes_20201215_082810-7f0de59a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_769x769_80k_cityscapes/dmnet_r101-d8_769x769_80k_cityscapes-20201215_082810.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                   | download                                                                                                                                                                                                                                                                                                                           |
| ------ | -------- | --------- | ------: | -------- | -------------- | ------ | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DMNet  | R-50-D8  | 512x512   |   80000 | 9.4      | 20.95          | V100   | 42.37 |         43.62 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r50-d8_4xb4-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x512_80k_ade20k/dmnet_r50-d8_512x512_80k_ade20k_20201215_144744-f89092a6.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x512_80k_ade20k/dmnet_r50-d8_512x512_80k_ade20k-20201215_144744.log.json)         |
| DMNet  | R-101-D8 | 512x512   |   80000 | 13.0     | 13.88          | V100   | 45.34 |         46.13 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r101-d8_4xb4-80k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x512_80k_ade20k/dmnet_r101-d8_512x512_80k_ade20k_20201215_104812-bfa45311.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x512_80k_ade20k/dmnet_r101-d8_512x512_80k_ade20k-20201215_104812.log.json)     |
| DMNet  | R-50-D8  | 512x512   |  160000 | -        | -              | V100   | 43.15 |         44.17 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r50-d8_4xb4-160k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x512_160k_ade20k/dmnet_r50-d8_512x512_160k_ade20k_20201215_115313-025ab3f9.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x512_160k_ade20k/dmnet_r50-d8_512x512_160k_ade20k-20201215_115313.log.json)     |
| DMNet  | R-101-D8 | 512x512   |  160000 | -        | -              | V100   | 45.42 |         46.76 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dmnet/dmnet_r101-d8_4xb4-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x512_160k_ade20k/dmnet_r101-d8_512x512_160k_ade20k_20201215_111145-a0bc02ef.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x512_160k_ade20k/dmnet_r101-d8_512x512_160k_ade20k-20201215_111145.log.json) |

## Citation

```bibtex
@InProceedings{He_2019_ICCV,
author = {He, Junjun and Deng, Zhongying and Qiao, Yu},
title = {Dynamic Multi-Scale Filters for Semantic Segmentation},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
