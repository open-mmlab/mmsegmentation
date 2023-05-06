# DDRNet

> [Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](http://arxiv.org/abs/2101.06085)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/ydhongHIT/DDRNet">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

Semantic segmentation is a key technology for autonomous vehicles to understand the surrounding scenes. The appealing performances of contemporary models usually come at the expense of heavy computations and lengthy inference time, which is intolerable for self-driving. Using light-weight architectures (encoder-decoder or two-pathway) or reasoning on low-resolution images, recent methods realize very fast scene parsing, even running at more than 100 FPS on a single 1080Ti GPU. However, there is still a signiﬁcant gap in performance between these real-time methods and the models based on dilation backbones. To tackle this problem, we proposed a family of efﬁcient backbones specially designed for real-time semantic segmentation. The proposed deep dual-resolution networks (DDRNets) are composed of two deep branches between which multiple bilateral fusions are performed. Additionally, we design a new contextual information extractor named Deep Aggregation Pyramid Pooling Module (DAPPM) to enlarge effective receptive ﬁelds and fuse multi-scale context based on low-resolution feature maps. Our method achieves a new state-of-the-art trade-off between accuracy and speed on both Cityscapes and CamVid dataset. In particular, on a single 2080Ti GPU, DDRNet-23-slim yields 77.4% mIoU at 102 FPS on Cityscapes test set and 74.7% mIoU at 230 FPS on CamVid test set. With widely used test augmentation, our method is superior to most state-of-the-art models and requires much less computation. Codes and trained models are available online.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/ydhongHIT/DDRNet/main/figs/DDRNet_seg.png" width="80%"/>
</div>

## Results and models

### Cityscapes

| Method | Backbone      | Crop Size | Lr schd | Mem(GB) | Inf time(fps) | Device | mIoU  | mIoU(ms+flip) | config                                                                                                                                    | download                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------ | ------------- | --------- | ------- | ------- | ------------- | ------ | ----- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DDRNet | DDRNet23-slim | 1024x1024 | 120000  | 1.70    | 85.85         | A100   | 77.84 | 80.15         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312.json) |
| DDRNet | DDRNet23      | 1024x1024 | 120000  | 7.26    | 33.41         | A100   | 79.99 | 81.71         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py)      | [model](https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230425_162633-81601db0.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230425_162633.json)                     |

## Notes

The pretrained weights in config files are converted from [the official repo](https://github.com/ydhongHIT/DDRNet#pretrained-models).

## Citation

```bibtex
@article{pan2022deep,
  title={Deep Dual-Resolution Networks for Real-Time and Accurate Semantic Segmentation of Traffic Scenes},
  author={Pan, Huihui and Hong, Yuanduo and Sun, Weichao and Jia, Yisong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2022},
  publisher={IEEE}
}
```
