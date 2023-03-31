# Fast-SCNN

> [Fast-SCNN for Semantic Segmentation](https://arxiv.org/abs/1902.04502)

## Introduction

<!-- [ALGORITHM] -->

<a href="">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/fast_scnn.py#L272">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The encoder-decoder framework is state-of-the-art for offline semantic image segmentation. Since the rise in autonomous systems, real-time computation is increasingly desirable. In this paper, we introduce fast segmentation convolutional neural network (Fast-SCNN), an above real-time semantic segmentation model on high resolution image data (1024x2048px) suited to efficient computation on embedded devices with low memory. Building on existing two-branch methods for fast segmentation, we introduce our \`learning to downsample' module which computes low-level features for multiple resolution branches simultaneously. Our network combines spatial detail at high resolution with deep features extracted at lower resolution, yielding an accuracy of 68.0% mean intersection over union at 123.5 frames per second on Cityscapes. We also show that large scale pre-training is unnecessary. We thoroughly validate our metric in experiments with ImageNet pre-training and the coarse labeled data of Cityscapes. Finally, we show even faster computation with competitive results on subsampled inputs, without any network modifications.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142901444-705b4ff4-6d1e-409b-899a-37bf3a6b69ce.png" width="80%"/>
</div>

## Results and models

### Cityscapes

| Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                       | download                                                                                                                                                                                                                                                                                                                                               |
| -------- | -------- | --------- | ------: | -------- | -------------- | ------ | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FastSCNN | FastSCNN | 512x1024  |  160000 | 3.3      | 56.45          | V100   | 70.96 | 72.65         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/fastscnn/fast_scnn_8xb4-160k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853.log.json) |

## Citation

```bibtex
@article{poudel2019fast,
  title={Fast-scnn: Fast semantic segmentation network},
  author={Poudel, Rudra PK and Liwicki, Stephan and Cipolla, Roberto},
  journal={arXiv preprint arXiv:1902.04502},
  year={2019}
}
```
