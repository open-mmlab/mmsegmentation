# EncNet

[Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/zhanghang1989/PyTorch-Encoding">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/enc_head.py#L63">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Recent work has made significant progress in improving spatial resolution for pixelwise labeling with Fully Convolutional Network (FCN) framework by employing Dilated/Atrous convolution, utilizing multi-scale features and refining boundaries. In this paper, we explore the impact of global contextual information in semantic segmentation by introducing the Context Encoding Module, which captures the semantic context of scenes and selectively highlights class-dependent featuremaps. The proposed Context Encoding Module significantly improves semantic segmentation results with only marginal extra computation cost over FCN. Our approach has achieved new state-of-the-art results 51.7% mIoU on PASCAL-Context, 85.9% mIoU on PASCAL VOC 2012. Our single model achieves a final score of 0.5567 on ADE20K test set, which surpass the winning entry of COCO-Place Challenge in 2017. In addition, we also explore how the Context Encoding Module can improve the feature representation of relatively shallow networks for the image classification on CIFAR-10 dataset. Our 14 layer network has achieved an error rate of 3.45%, which is comparable with state-of-the-art approaches with over 10 times more layers. The source code for the complete system are publicly available.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142901276-b364fbbf-3bdb-4000-9d31-b9a135e30935.png" width="70%"/>
</div>

## Citation

```bibtex
@InProceedings{Zhang_2018_CVPR,
author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
title = {Context Encoding for Semantic Segmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                            | download                                                                                                                                                                                                                                                                                                                                                 |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EncNet | R-50-D8  | 512x1024  |   40000 | 8.6      | 4.58           | 75.67 |         77.08 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r50-d8_4xb2-40k_cityscapes-512x1024.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_40k_cityscapes/encnet_r50-d8_512x1024_40k_cityscapes_20200621_220958-68638a47.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_40k_cityscapes/encnet_r50-d8_512x1024_40k_cityscapes-20200621_220958.log.json)     |
| EncNet | R-101-D8 | 512x1024  |   40000 | 12.1     | 2.66           | 75.81 |         77.21 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r101-d8_4xb2-40k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x1024_40k_cityscapes/encnet_r101-d8_512x1024_40k_cityscapes_20200621_220933-35e0a3e8.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x1024_40k_cityscapes/encnet_r101-d8_512x1024_40k_cityscapes-20200621_220933.log.json) |
| EncNet | R-50-D8  | 769x769   |   40000 | 9.8      | 1.82           | 76.24 |         77.85 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r50-d8_4xb2-40k_cityscapes-769x769.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_769x769_40k_cityscapes/encnet_r50-d8_769x769_40k_cityscapes_20200621_220958-3bcd2884.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_769x769_40k_cityscapes/encnet_r50-d8_769x769_40k_cityscapes-20200621_220958.log.json)         |
| EncNet | R-101-D8 | 769x769   |   40000 | 13.7     | 1.26           | 74.25 |         76.25 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r101-d8_4xb2-40k_cityscapes-769x769.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_769x769_40k_cityscapes/encnet_r101-d8_769x769_40k_cityscapes_20200621_220933-2fafed55.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_769x769_40k_cityscapes/encnet_r101-d8_769x769_40k_cityscapes-20200621_220933.log.json)     |
| EncNet | R-50-D8  | 512x1024  |   80000 | -        | -              | 77.94 |         79.13 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r50-d8_4xb2-80k_cityscapes-512x1024.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_80k_cityscapes/encnet_r50-d8_512x1024_80k_cityscapes_20200622_003554-fc5c5624.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_80k_cityscapes/encnet_r50-d8_512x1024_80k_cityscapes-20200622_003554.log.json)     |
| EncNet | R-101-D8 | 512x1024  |   80000 | -        | -              | 78.55 |         79.47 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r101-d8_4xb2-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x1024_80k_cityscapes/encnet_r101-d8_512x1024_80k_cityscapes_20200622_003555-1de64bec.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x1024_80k_cityscapes/encnet_r101-d8_512x1024_80k_cityscapes-20200622_003555.log.json) |
| EncNet | R-50-D8  | 769x769   |   80000 | -        | -              | 77.44 |         78.72 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r50-d8_4xb2-80k_cityscapes-769x769.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_769x769_80k_cityscapes/encnet_r50-d8_769x769_80k_cityscapes_20200622_003554-55096dcb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_769x769_80k_cityscapes/encnet_r50-d8_769x769_80k_cityscapes-20200622_003554.log.json)         |
| EncNet | R-101-D8 | 769x769   |   80000 | -        | -              | 76.10 |         76.97 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r101-d8_4xb2-80k_cityscapes-769x769.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_769x769_80k_cityscapes/encnet_r101-d8_769x769_80k_cityscapes_20200622_003555-470ef79d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_769x769_80k_cityscapes/encnet_r101-d8_769x769_80k_cityscapes-20200622_003555.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                        | download                                                                                                                                                                                                                                                                                                                                 |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EncNet | R-50-D8  | 512x512   |   80000 | 10.1     | 22.81          | 39.53 |         41.17 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r50-d8_4xb4-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x512_80k_ade20k/encnet_r50-d8_512x512_80k_ade20k_20200622_042412-44b46b04.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x512_80k_ade20k/encnet_r50-d8_512x512_80k_ade20k-20200622_042412.log.json)         |
| EncNet | R-101-D8 | 512x512   |   80000 | 13.6     | 14.87          | 42.11 |         43.61 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r101-d8_4xb4-80k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x512_80k_ade20k/encnet_r101-d8_512x512_80k_ade20k_20200622_101128-dd35e237.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x512_80k_ade20k/encnet_r101-d8_512x512_80k_ade20k-20200622_101128.log.json)     |
| EncNet | R-50-D8  | 512x512   |  160000 | -        | -              | 40.10 |         41.71 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r50-d8_4xb4-160k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x512_160k_ade20k/encnet_r50-d8_512x512_160k_ade20k_20200622_101059-b2db95e0.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x512_160k_ade20k/encnet_r50-d8_512x512_160k_ade20k-20200622_101059.log.json)     |
| EncNet | R-101-D8 | 512x512   |  160000 | -        | -              | 42.61 |         44.01 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/encnet/encnet_r101-d8_4xb4-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x512_160k_ade20k/encnet_r101-d8_512x512_160k_ade20k_20200622_073348-7989641f.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r101-d8_512x512_160k_ade20k/encnet_r101-d8_512x512_160k_ade20k-20200622_073348.log.json) |
