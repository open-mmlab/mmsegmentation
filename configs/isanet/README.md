# ISANet

[Interlaced Sparse Self-Attention for Semantic Segmentation](https://arxiv.org/abs/1907.12273)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/openseg-group/openseg.pytorch">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/decode_heads/isa_head.py#L58">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

In this paper, we present a so-called interlaced sparse self-attention approach to improve the efficiency of the \\emph{self-attention} mechanism for semantic segmentation. The main idea is that we factorize the dense affinity matrix as the product of two sparse affinity matrices. There are two successive attention modules each estimating a sparse affinity matrix. The first attention module is used to estimate the affinities within a subset of positions that have long spatial interval distances and the second attention module is used to estimate the affinities within a subset of positions that have short spatial interval distances. These two attention modules are designed so that each position is able to receive the information from all the other positions. In contrast to the original self-attention module, our approach decreases the computation and memory complexity substantially especially when processing high-resolution feature maps. We empirically verify the effectiveness of our approach on six challenging semantic segmentation benchmarks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142901868-03d80da4-b9c0-4df9-8509-5f684ba9dadc.png" width="80%"/>
</div>

## Citation

```bibetex
@article{huang2019isa,
  title={Interlaced Sparse Self-Attention for Semantic Segmentation},
  author={Huang, Lang and Yuan, Yuhui and Guo, Jianyuan and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal={arXiv preprint arXiv:1907.12273},
  year={2019}
}
```

The technical report above is also presented at:

```bibetex
@article{yuan2021ocnet,
  title={OCNet: Object Context for Semantic Segmentation},
  author={Yuan, Yuhui and Huang, Lang and Guo, Jianyuan and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal={International Journal of Computer Vision},
  pages={1--24},
  year={2021},
  publisher={Springer}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                            config | download                                                                                                                                                                                                                                                                                                                                                 |
| ------ | -------- | --------- | ------- | -------: | -------------- | ----- | ------------: | --------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ISANet | R-50-D8  | 512x1024  | 40000   |    5.869 | 2.91           | 78.49 |         79.44 |  [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r50-d8_4xb2-40k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x1024_40k_cityscapes/isanet_r50-d8_512x1024_40k_cityscapes_20210901_054739-981bd763.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x1024_40k_cityscapes/isanet_r50-d8_512x1024_40k_cityscapes_20210901_054739.log.json)     |
| ISANet | R-50-D8  | 512x1024  | 80000   |    5.869 | 2.91           | 78.68 |         80.25 |  [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r50-d8_4xb2-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x1024_80k_cityscapes/isanet_r50-d8_512x1024_80k_cityscapes_20210901_074202-89384497.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x1024_80k_cityscapes/isanet_r50-d8_512x1024_80k_cityscapes_20210901_074202.log.json)     |
| ISANet | R-50-D8  | 769x769   | 40000   |    6.759 | 1.54           | 78.70 |         80.28 |   [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r50-d8_4xb2-40k_cityscapes-769x769.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_769x769_40k_cityscapes/isanet_r50-d8_769x769_40k_cityscapes_20210903_050200-4ae7e65b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_769x769_40k_cityscapes/isanet_r50-d8_769x769_40k_cityscapes_20210903_050200.log.json)         |
| ISANet | R-50-D8  | 769x769   | 80000   |    6.759 | 1.54           | 79.29 |         80.53 |   [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r50-d8_4xb2-80k_cityscapes-769x769.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_769x769_80k_cityscapes/isanet_r50-d8_769x769_80k_cityscapes_20210903_101126-99b54519.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_769x769_80k_cityscapes/isanet_r50-d8_769x769_80k_cityscapes_20210903_101126.log.json)         |
| ISANet | R-101-D8 | 512x1024  | 40000   |    9.425 | 2.35           | 79.58 |         81.05 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r101-d8_4xb2-40k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x1024_40k_cityscapes/isanet_r101-d8_512x1024_40k_cityscapes_20210901_145553-293e6bd6.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x1024_40k_cityscapes/isanet_r101-d8_512x1024_40k_cityscapes_20210901_145553.log.json) |
| ISANet | R-101-D8 | 512x1024  | 80000   |    9.425 | 2.35           | 80.32 |         81.58 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r101-d8_4xb2-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x1024_80k_cityscapes/isanet_r101-d8_512x1024_80k_cityscapes_20210901_145243-5b99c9b2.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x1024_80k_cityscapes/isanet_r101-d8_512x1024_80k_cityscapes_20210901_145243.log.json) |
| ISANet | R-101-D8 | 769x769   | 40000   |   10.815 | 0.92           | 79.68 |         80.95 |  [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r101-d8_4xb2-40k_cityscapes-769x769.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_769x769_40k_cityscapes/isanet_r101-d8_769x769_40k_cityscapes_20210903_111320-509e7224.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_769x769_40k_cityscapes/isanet_r101-d8_769x769_40k_cityscapes_20210903_111320.log.json)     |
| ISANet | R-101-D8 | 769x769   | 80000   |   10.815 | 0.92           | 80.61 |         81.59 |  [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r101-d8_4xb2-80k_cityscapes-769x769.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_769x769_80k_cityscapes/isanet_r101-d8_769x769_80k_cityscapes_20210903_111319-24f71dfa.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_769x769_80k_cityscapes/isanet_r101-d8_769x769_80k_cityscapes_20210903_111319.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                        config | download                                                                                                                                                                                                                                                                                                                                 |
| ------ | -------- | --------- | ------- | -------: | -------------- | ----- | ------------: | ----------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ISANet | R-50-D8  | 512x512   | 80000   |      9.0 | 22.55          | 41.12 |         42.35 |   [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r50-d8_4xb4-80k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_80k_ade20k/isanet_r50-d8_512x512_80k_ade20k_20210903_124557-6ed83a0c.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_80k_ade20k/isanet_r50-d8_512x512_80k_ade20k_20210903_124557.log.json)         |
| ISANet | R-50-D8  | 512x512   | 160000  |      9.0 | 22.55          | 42.59 |         43.07 |  [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r50-d8_4xb4-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_160k_ade20k/isanet_r50-d8_512x512_160k_ade20k_20210903_104850-f752d0a3.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_160k_ade20k/isanet_r50-d8_512x512_160k_ade20k_20210903_104850.log.json)     |
| ISANet | R-101-D8 | 512x512   | 80000   |   12.562 | 10.56          | 43.51 |         44.38 |  [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r101-d8_4xb4-80k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_80k_ade20k/isanet_r101-d8_512x512_80k_ade20k_20210903_162056-68b235c2.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_80k_ade20k/isanet_r101-d8_512x512_80k_ade20k_20210903_162056.log.json)     |
| ISANet | R-101-D8 | 512x512   | 160000  |   12.562 | 10.56          | 43.80 |          45.4 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r101-d8_4xb4-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_160k_ade20k/isanet_r101-d8_512x512_160k_ade20k_20210903_211431-a7879dcd.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_160k_ade20k/isanet_r101-d8_512x512_160k_ade20k_20210903_211431.log.json) |

### Pascal VOC 2012 + Aug

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                         config | download                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ------- | -------: | -------------- | ----- | ------------: | -----------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ISANet | R-50-D8  | 512x512   | 20000   |      5.9 | 23.08          | 76.78 |         77.79 |  [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r50-d8_4xb4-20k_voc12aug-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_20k_voc12aug/isanet_r50-d8_512x512_20k_voc12aug_20210901_164838-79d59b80.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_20k_voc12aug/isanet_r50-d8_512x512_20k_voc12aug_20210901_164838.log.json)     |
| ISANet | R-50-D8  | 512x512   | 40000   |      5.9 | 23.08          | 76.20 |         77.22 |  [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r50-d8_4xb4-40k_voc12aug-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_40k_voc12aug/isanet_r50-d8_512x512_40k_voc12aug_20210901_151349-7d08a54e.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_40k_voc12aug/isanet_r50-d8_512x512_40k_voc12aug_20210901_151349.log.json)     |
| ISANet | R-101-D8 | 512x512   | 20000   |    9.465 | 7.42           | 78.46 |         79.16 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r101-d8_4xb4-20k_voc12aug-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_20k_voc12aug/isanet_r101-d8_512x512_20k_voc12aug_20210901_115805-3ccbf355.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_20k_voc12aug/isanet_r101-d8_512x512_20k_voc12aug_20210901_115805.log.json) |
| ISANet | R-101-D8 | 512x512   | 40000   |    9.465 | 7.42           | 78.12 |         79.04 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/isanet/isanet_r101-d8_4xb4-40k_voc12aug-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_40k_voc12aug/isanet_r101-d8_512x512_40k_voc12aug_20210901_145814-bc71233b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_40k_voc12aug/isanet_r101-d8_512x512_40k_voc12aug_20210901_145814.log.json) |
