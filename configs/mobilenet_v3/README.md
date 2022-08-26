# MobileNetV3

[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

## Introduction

<!-- [BACKBONE] -->

<!-- [ALGORITHM] -->

<a href="https://github.com/tensorflow/models/tree/master/research/deeplab">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/mobilenet_v3.py#L15">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2% more accurate on ImageNet classification while reducing latency by 15% compared to MobileNetV2. MobileNetV3-Small is 4.6% more accurate while reducing latency by 5% compared to MobileNetV2. MobileNetV3-Large detection is 25% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142902036-3dc2e0c0-d475-4816-b1ac-961836b41f5c.png" width="60%"/>
</div>

## Citation

```bibtex
@inproceedings{Howard_2019_ICCV,
  title={Searching for MobileNetV3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  pages={1314-1324},
  month={October},
  year={2019},
  doi={10.1109/ICCV.2019.00140}}
}
```

## Results and models

### Cityscapes

| Method | Backbone           | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------ | ------------------ | --------- | ------: | -------: | -------------- | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LRASPP | M-V3-D8            | 512x1024  |  320000 |      8.9 | 15.22          | 69.54 | 70.89         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/mobilenet_v3/mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes/lraspp_m-v3-d8_512x1024_320k_cityscapes_20201224_220337-cfe8fb07.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes/lraspp_m-v3-d8_512x1024_320k_cityscapes-20201224_220337.log.json)                                     |
| LRASPP | M-V3-D8 (scratch)  | 512x1024  |  320000 |      8.9 | 14.77          | 67.87 | 69.78         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/mobilenet_v3/mobilenet-v3-d8-scratch_lraspp_4xb4-320k_cityscapes-512x1024.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes/lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes_20201224_220337-9f29cd72.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes/lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes-20201224_220337.log.json)     |
| LRASPP | M-V3s-D8           | 512x1024  |  320000 |      5.3 | 23.64          | 64.11 | 66.42         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/mobilenet_v3/mobilenet-v3-d8-s_lraspp_4xb4-320k_cityscapes-512x1024.py)         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3s-d8_512x1024_320k_cityscapes/lraspp_m-v3s-d8_512x1024_320k_cityscapes_20201224_223935-61565b34.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3s-d8_512x1024_320k_cityscapes/lraspp_m-v3s-d8_512x1024_320k_cityscapes-20201224_223935.log.json)                                 |
| LRASPP | M-V3s-D8 (scratch) | 512x1024  |  320000 |      5.3 | 24.50          | 62.74 | 65.01         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/mobilenet_v3/mobilenet-v3-d8-scratch-s_lraspp_4xb4-320k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3s-d8_scratch_512x1024_320k_cityscapes/lraspp_m-v3s-d8_scratch_512x1024_320k_cityscapes_20201224_223935-03daeabb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3s-d8_scratch_512x1024_320k_cityscapes/lraspp_m-v3s-d8_scratch_512x1024_320k_cityscapes-20201224_223935.log.json) |
