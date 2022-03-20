# DNLNet

[Disentangled Non-Local Neural Networks](https://arxiv.org/abs/2006.06668)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/yinmh17/DNL-Semantic-Segmentation">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/dnl_head.py#L88">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The non-local block is a popular module for strengthening the context modeling ability of a regular convolutional neural network. This paper first studies the non-local block in depth, where we find that its attention computation can be split into two terms, a whitened pairwise term accounting for the relationship between two pixels and a unary term representing the saliency of every pixel. We also observe that the two terms trained alone tend to model different visual clues, e.g. the whitened pairwise term learns within-region relationships while the unary term learns salient boundaries. However, the two terms are tightly coupled in the non-local block, which hinders the learning of each. Based on these findings, we present the disentangled non-local block, where the two terms are decoupled to facilitate learning for both terms. We demonstrate the effectiveness of the decoupled design on various tasks, such as semantic segmentation on Cityscapes, ADE20K and PASCAL Context, object detection on COCO, and action recognition on Kinetics.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142900944-b8d93301-d2ce-488e-a461-b0813f96be49.png" width="70%"/>
</div>

## Citation

This example is to reproduce ["Disentangled Non-Local Neural Networks"](https://arxiv.org/abs/2006.06668) for semantic segmentation. It is still in progress.

## Citation

```bibtex
@misc{yin2020disentangled,
    title={Disentangled Non-Local Neural Networks},
    author={Minghao Yin and Zhuliang Yao and Yue Cao and Xiu Li and Zheng Zhang and Stephen Lin and Han Hu},
    year={2020},
    booktitle={ECCV}
}
```

## Results and models (in progress)

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                   | download                                                                                                                                                                                                                                                                                                                                         |
| ------ | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DNLNet    | R-50-D8  | 512x1024  |   40000 |      7.3 | 2.56           | 78.61 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r50-d8_512x1024_40k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x1024_40k_cityscapes/dnl_r50-d8_512x1024_40k_cityscapes_20200904_233629-53d4ea93.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x1024_40k_cityscapes/dnl_r50-d8_512x1024_40k_cityscapes-20200904_233629.log.json)     |
| DNLNet    | R-101-D8 | 512x1024  |   40000 |     10.9 | 1.96           | 78.31 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r101-d8_512x1024_40k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x1024_40k_cityscapes/dnl_r101-d8_512x1024_40k_cityscapes_20200904_233629-9928ffef.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x1024_40k_cityscapes/dnl_r101-d8_512x1024_40k_cityscapes-20200904_233629.log.json) |
| DNLNet    | R-50-D8  | 769x769   |   40000 |      9.2 | 1.50           | 78.44 | 80.27         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r50-d8_769x769_40k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_769x769_40k_cityscapes/dnl_r50-d8_769x769_40k_cityscapes_20200820_232206-0f283785.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_769x769_40k_cityscapes/dnl_r50-d8_769x769_40k_cityscapes-20200820_232206.log.json)         |
| DNLNet    | R-101-D8 | 769x769   |   40000 |     12.6 | 1.02           | 76.39 | 77.77         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r101-d8_769x769_40k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_769x769_40k_cityscapes/dnl_r101-d8_769x769_40k_cityscapes_20200820_171256-76c596df.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_769x769_40k_cityscapes/dnl_r101-d8_769x769_40k_cityscapes-20200820_171256.log.json)     |
| DNLNet    | R-50-D8  | 512x1024  |   80000 |        - | -              | 79.33 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r50-d8_512x1024_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x1024_80k_cityscapes/dnl_r50-d8_512x1024_80k_cityscapes_20200904_233629-58b2f778.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x1024_80k_cityscapes/dnl_r50-d8_512x1024_80k_cityscapes-20200904_233629.log.json)     |
| DNLNet    | R-101-D8 | 512x1024  |   80000 |        - | -              | 80.41 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r101-d8_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x1024_80k_cityscapes/dnl_r101-d8_512x1024_80k_cityscapes_20200904_233629-758e2dd4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x1024_80k_cityscapes/dnl_r101-d8_512x1024_80k_cityscapes-20200904_233629.log.json) |
| DNLNet    | R-50-D8  | 769x769   |   80000 |        - | -              | 79.36 | 80.70         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r50-d8_769x769_80k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_769x769_80k_cityscapes/dnl_r50-d8_769x769_80k_cityscapes_20200820_011925-366bc4c7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_769x769_80k_cityscapes/dnl_r50-d8_769x769_80k_cityscapes-20200820_011925.log.json)         |
| DNLNet    | R-101-D8 | 769x769   |   80000 |        - | -              | 79.41 | 80.68         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r101-d8_769x769_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_769x769_80k_cityscapes/dnl_r101-d8_769x769_80k_cityscapes_20200821_051111-95ff84ab.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_769x769_80k_cityscapes/dnl_r101-d8_769x769_80k_cityscapes-20200821_051111.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                               | download                                                                                                                                                                                                                                                                                                                         |
| ------ | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DNLNet    | R-50-D8  | 512x512   |   80000 |      8.8 | 20.66          | 41.76 | 42.99         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r50-d8_512x512_80k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_80k_ade20k/dnl_r50-d8_512x512_80k_ade20k_20200826_183354-1cf6e0c1.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_80k_ade20k/dnl_r50-d8_512x512_80k_ade20k-20200826_183354.log.json)         |
| DNLNet    | R-101-D8 | 512x512   |   80000 |     12.8 | 12.54          | 43.76 | 44.91         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r101-d8_512x512_80k_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x512_80k_ade20k/dnl_r101-d8_512x512_80k_ade20k_20200826_183354-d820d6ea.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x512_80k_ade20k/dnl_r101-d8_512x512_80k_ade20k-20200826_183354.log.json)     |
| DNLNet    | R-50-D8  | 512x512   |  160000 |        - | -              | 41.87 | 43.01         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r50-d8_512x512_160k_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_160k_ade20k/dnl_r50-d8_512x512_160k_ade20k_20200826_183350-37837798.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r50-d8_512x512_160k_ade20k/dnl_r50-d8_512x512_160k_ade20k-20200826_183350.log.json)     |
| DNLNet    | R-101-D8 | 512x512   |  160000 |        - | -              | 44.25 | 45.78         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet/dnl_r101-d8_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x512_160k_ade20k/dnl_r101-d8_512x512_160k_ade20k_20200826_183350-ed522c61.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dnlnet/dnl_r101-d8_512x512_160k_ade20k/dnl_r101-d8_512x512_160k_ade20k-20200826_183350.log.json) |
