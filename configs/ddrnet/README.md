# DDRNet

> [Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](http://arxiv.org/abs/2101.06085)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/ydhongHIT/DDRNet">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

Semantic segmentation is a key technology for autonomous vehicles to understand the surrounding scenes. The appealing performances of contemporary models usually come at the expense of heavy computations and lengthy inference time, which is intolerable for self-driving. Using light-weight architectures (encoder-decoder or two-pathway) or reasoning on low-resolution images, recent methods realize very fast scene parsing, even running at more than 100 FPS on a single 1080Ti GPU. However, there is still a signiﬁcant gap in performance between these real-time methods and the models based on dilation backbones. To tackle this problem, we proposed a family of efﬁcient backbones specially designed for real-time semantic segmentation. The proposed deep dual-resolution networks (DDRNets) are composed of two deep branches between which multiple bilateral fusions are performed. Additionally, we design a new contextual information extractor named Deep Aggregation Pyramid Pooling Module (DAPPM) to enlarge effective receptive ﬁelds and fuse multi-scale context based on low-resolution feature maps. Our method achieves a new state-of-the-art trade-off between accuracy and speed on both Cityscapes and CamVid dataset. In particular, on a single 2080Ti GPU, DDRNet-23-slim yields 77.4% mIoU at 102 FPS on Cityscapes test set and 74.7% mIoU at 230 FPS on CamVid test set. With widely used test augmentation, our method is superior to most state-of-the-art models and requires much less computation. Codes and trained models are available online.

<!-- [IMAGE] -->

<!-- <div align=center>
<img src="https://raw.githubusercontent.com/ydhongHIT/DDRNet/main/figs/DDRNet_seg.png" width="60%"/>
</div> -->

## Results and models

### Cityscapes

| Method | Backbone      | Crop Size | Lr schd | Mem(GB) | Inf time(fps) | Device   | mIoU  | mIoU(ms+flip) | config       | download     |
| ------ | ------------- | --------- | ------- | ------- | ------------- | -------- | ----- | ------------- | ------------ | ------------ |
| DDRNet | DDRNet23-slim | 1024x1024 | 120000  |         | 85.85         | RTX 8000 | 77.85 | 79.80         | [config](<>) | model \| log |
| DDRNet | DDRNet23      | 1024x1024 | 120000  |         | 33.41         | RTX 8000 | 79.53 | 80.98         | [config](<>) | model \| log |
| DDRNet | DDRNet39      | 1024x1024 | 120000  |         |               | RTX 8000 |       |               | [config](<>) | model \| log |

## Notes

The pretrained weights in config files are converted from [the official repo](https://github.com/ydhongHIT/DDRNet#pretrained-models).

## Citation

```bibtex
@misc{hong2021ddrnet,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  year={2021},
  eprint={2101.06085},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
}
```
