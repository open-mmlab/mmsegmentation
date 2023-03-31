# BiSeNetV2

> [Bisenet v2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/abs/2004.02147)

## Introduction

<!-- [ALGORITHM] -->

<a href="">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/backbones/bisenetv2.py#L545">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The low-level details and high-level semantics are both essential to the semantic segmentation task. However, to speed up the model inference, current approaches almost always sacrifice the low-level details, which leads to a considerable accuracy decrease. We propose to treat these spatial details and categorical semantics separately to achieve high accuracy and high efficiency for realtime semantic segmentation. To this end, we propose an efficient and effective architecture with a good trade-off between speed and accuracy, termed Bilateral Segmentation Network (BiSeNet V2). This architecture involves: (i) a Detail Branch, with wide channels and shallow layers to capture low-level details and generate high-resolution feature representation; (ii) a Semantic Branch, with narrow channels and deep layers to obtain high-level semantic context. The Semantic Branch is lightweight due to reducing the channel capacity and a fast-downsampling strategy. Furthermore, we design a Guided Aggregation Layer to enhance mutual connections and fuse both types of feature representation. Besides, a booster training strategy is designed to improve the segmentation performance without any extra inference cost. Extensive quantitative and qualitative evaluations demonstrate that the proposed architecture performs favourably against a few state-of-the-art real-time semantic segmentation approaches. Specifically, for a 2,048x1,024 input, we achieve 72.6% Mean IoU on the Cityscapes test set with a speed of 156 FPS on one NVIDIA GeForce GTX 1080 Ti card, which is significantly faster than existing methods, yet we achieve better segmentation accuracy.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142898966-ec4a81da-b4b0-41ee-b083-1d964582c18a.png" width="70%"/>
</div>

## Results and models

### Cityscapes

| Method    | Backbone         | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                                  | download                                                                                                                                                                                                                                                                                                                                                                                               |
| --------- | ---------------- | --------- | ------: | -------- | -------------- | ------ | ----: | ------------: | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BiSeNetV2 | BiSeNetV2        | 1024x1024 |  160000 | 7.64     | 31.77          | V100   | 73.21 |         75.74 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/bisenetv2/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py)      | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551.log.json)                     |
| BiSeNetV2 | BiSeNetV2 (OHEM) | 1024x1024 |  160000 | 7.64     | -              | V100   | 73.57 |         75.80 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/bisenetv2/bisenetv2_fcn_4xb4-ohem-160k_cityscapes-1024x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes_20210902_112947-5f8103b4.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes_20210902_112947.log.json) |
| BiSeNetV2 | BiSeNetV2 (4x8)  | 1024x1024 |  160000 | 15.05    | -              | V100   | 75.76 |         77.79 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/bisenetv2/bisenetv2_fcn_4xb8-160k_cityscapes-1024x1024.py)      | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032.log.json)                     |
| BiSeNetV2 | BiSeNetV2 (FP16) | 1024x1024 |  160000 | 5.77     | 36.65          | V100   | 73.07 |         75.13 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/bisenetv2/bisenetv2_fcn_4xb4-amp-160k_cityscapes-1024x1024.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942-b979777b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942.log.json) |

Note:

- `OHEM` means Online Hard Example Mining (OHEM) is adopted in training.
- `FP16` means Mixed Precision (FP16) is adopted in training.
- `4x8` means 4 GPUs with 8 samples per GPU in training.

## Citation

```bibtex
@article{yu2021bisenet,
  title={Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation},
  author={Yu, Changqian and Gao, Changxin and Wang, Jingbo and Yu, Gang and Shen, Chunhua and Sang, Nong},
  journal={International Journal of Computer Vision},
  pages={1--18},
  year={2021},
  publisher={Springer}
}
```
