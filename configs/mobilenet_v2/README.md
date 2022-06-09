# MobileNetV2

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/tensorflow/models/tree/master/research/deeplab">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/mobilenet_v2.py#L14">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3.
The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142901935-fa22700e-4b77-477f-90b9-334a4197506f.png" width="50%"/>
</div>

## Citation

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

## Results and models

### Cityscapes

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                   | download                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------- | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FCN        | M-V2-D8  | 512x1024  |   80000 |      3.4 | 14.2           | 61.54 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes/fcn_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-d24c28c1.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes/fcn_m-v2-d8_512x1024_80k_cityscapes-20200825_124817.log.json)                                         |
| PSPNet     | M-V2-D8  | 512x1024  |   80000 |      3.6 | 11.2           | 70.23 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes/pspnet_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-19e81d51.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes/pspnet_m-v2-d8_512x1024_80k_cityscapes-20200825_124817.log.json)                             |
| DeepLabV3  | M-V2-D8  | 512x1024  |   80000 |      3.9 | 8.4            | 73.84 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes/deeplabv3_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-bef03590.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes/deeplabv3_m-v2-d8_512x1024_80k_cityscapes-20200825_124836.log.json)                 |
| DeepLabV3+ | M-V2-D8  | 512x1024  |   80000 |      5.1 | 8.4            | 75.20 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-d256dd4b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes-20200825_124836.log.json) |

### ADE20K

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                               | download                                                                                                                                                                                                                                                                                                                                                                         |
| ---------- | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN        | M-V2-D8  | 512x512   |  160000 |      6.5 | 64.4           | 19.71 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k/fcn_m-v2-d8_512x512_160k_ade20k_20200825_214953-c40e1095.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k/fcn_m-v2-d8_512x512_160k_ade20k-20200825_214953.log.json)                                         |
| PSPNet     | M-V2-D8  | 512x512   |  160000 |      6.5 | 57.7           | 29.68 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/pspnet_m-v2-d8_512x512_160k_ade20k.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x512_160k_ade20k/pspnet_m-v2-d8_512x512_160k_ade20k_20200825_214953-f5942f7a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x512_160k_ade20k/pspnet_m-v2-d8_512x512_160k_ade20k-20200825_214953.log.json)                             |
| DeepLabV3  | M-V2-D8  | 512x512   |  160000 |      6.8 | 39.9           | 34.08 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k/deeplabv3_m-v2-d8_512x512_160k_ade20k-20200825_223255.log.json)                 |
| DeepLabV3+ | M-V2-D8  | 512x512   |  160000 |      8.2 | 43.1           | 34.02 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3plus_m-v2-d8_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x512_160k_ade20k/deeplabv3plus_m-v2-d8_512x512_160k_ade20k_20200825_223255-465a01d4.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x512_160k_ade20k/deeplabv3plus_m-v2-d8_512x512_160k_ade20k-20200825_223255.log.json) |
