# EMANet

> [Expectation-Maximization Attention Networks for Semantic Segmentation](https://arxiv.org/abs/1907.13426)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://xialipku.github.io/EMANet">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/ema_head.py#L80">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Self-attention mechanism has been widely used for various tasks. It is designed to compute the representation of each position by a weighted sum of the features at all positions. Thus, it can capture long-range relations for computer vision tasks. However, it is computationally consuming. Since the attention maps are computed w.r.t all other positions. In this paper, we formulate the attention mechanism into an expectation-maximization manner and iteratively estimate a much more compact set of bases upon which the attention maps are computed. By a weighted summation upon these bases, the resulting representation is low-rank and deprecates noisy information from the input. The proposed Expectation-Maximization Attention (EMA) module is robust to the variance of input and is also friendly in memory and computation. Moreover, we set up the bases maintenance and normalization methods to stabilize its training procedure. We conduct extensive experiments on popular semantic segmentation benchmarks including PASCAL VOC, PASCAL Context and COCO Stuff, on which we set new records.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142901186-7bfe15e2-805a-420e-81b0-74f214f20a36.png" width="80%"/>
</div>

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                           | download                                                                                                                                                                                                                                                                                                                                                 |
| ------ | -------- | --------- | ------: | -------: | -------------- | ------ | ----: | ------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EMANet | R-50-D8  | 512x1024  |   80000 |      5.4 | 4.58           | V100   | 77.59 | 79.44         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/emanet/eemanet_r50-d8_4xb2-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/emanet/emanet_r50-d8_512x1024_80k_cityscapes/emanet_r50-d8_512x1024_80k_cityscapes_20200901_100301-c43fcef1.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/emanet/emanet_r50-d8_512x1024_80k_cityscapes/emanet_r50-d8_512x1024_80k_cityscapes-20200901_100301.log.json)     |
| EMANet | R-101-D8 | 512x1024  |   80000 |      6.2 | 2.87           | V100   | 79.10 | 81.21         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/emanet/emanet_r101-d8_4xb2-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/emanet/emanet_r101-d8_512x1024_80k_cityscapes/emanet_r101-d8_512x1024_80k_cityscapes_20200901_100301-2d970745.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/emanet/emanet_r101-d8_512x1024_80k_cityscapes/emanet_r101-d8_512x1024_80k_cityscapes-20200901_100301.log.json) |
| EMANet | R-50-D8  | 769x769   |   80000 |      8.9 | 1.97           | V100   | 79.33 | 80.49         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/emanet/emanet_r50-d8_4xb2-80k_cityscapes-769x769.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/emanet/emanet_r50-d8_769x769_80k_cityscapes/emanet_r50-d8_769x769_80k_cityscapes_20200901_100301-16f8de52.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/emanet/emanet_r50-d8_769x769_80k_cityscapes/emanet_r50-d8_769x769_80k_cityscapes-20200901_100301.log.json)         |
| EMANet | R-101-D8 | 769x769   |   80000 |     10.1 | 1.22           | V100   | 79.62 | 81.00         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/emanet/emanet_r101-d8_4xb2-80k_cityscapes-769x769.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/emanet/emanet_r101-d8_769x769_80k_cityscapes/emanet_r101-d8_769x769_80k_cityscapes_20200901_100301-47a324ce.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/emanet/emanet_r101-d8_769x769_80k_cityscapes/emanet_r101-d8_769x769_80k_cityscapes-20200901_100301.log.json)     |

## Citation

```bibtex
@inproceedings{li2019expectation,
  title={Expectation-maximization attention networks for semantic segmentation},
  author={Li, Xia and Zhong, Zhisheng and Wu, Jianlong and Yang, Yibo and Lin, Zhouchen and Liu, Hong},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9167--9176},
  year={2019}
}
```
