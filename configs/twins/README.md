# Twins

[Twins: Revisiting the Design of Spatial Attention in Vision Transformers](https://arxiv.org/pdf/2104.13840.pdf)

## Introduction

<!-- [BACKBONE] -->

<a href = "https://github.com/Meituan-AutoML/Twins">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.20.0/mmseg/models/backbones/twins.py#L352">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully-devised yet simple spatial attention mechanism performs favourably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins-PCPVT and Twins-SVT. Our proposed architectures are highly-efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks, including image level classification as well as dense detection and segmentation. The simplicity and strong performance suggest that our proposed architectures may serve as stronger backbones for many vision tasks. Our code is released at [this https URL](https://github.com/Meituan-AutoML/Twins).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/145021310-57826cf5-5e03-4c7c-9081-ffa744bdae27.png" width="80%"/>
</div>

## Citation

```bibtex
@article{chu2021twins,
  title={Twins: Revisiting spatial attention design in vision transformers},
  author={Chu, Xiangxiang and Tian, Zhi and Wang, Yuqing and Zhang, Bo and Ren, Haibing and Wei, Xiaolin and Xia, Huaxia and Shen, Chunhua},
  journal={arXiv preprint arXiv:2104.13840},
  year={2021}altgvt
}
```

## Usage

We have provided pretrained models converted from [official repo](https://github.com/Meituan-AutoML/Twins).

If you want to convert keys on your own to use official repositories' pre-trained models, we also provide a script [`twins2mmseg.py`](../../tools/model_converters/twins2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/Meituan-AutoML/Twins) to MMSegmentation style.

```shell
python tools/model_converters/twins2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH} ${MODEL_TYPE}
```

This script convert `pcpvt` or `svt` pretrained model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

For example,

```shell
python tools/model_converters/twins2mmseg.py ./alt_gvt_base.pth ./pretrained/alt_gvt_base.pth svt
```

## Results and models

### ADE20K

| Method              | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config                                                                                                                                 | download                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------- | -------- | --------- | ------- | -------- | -------------- | ----- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Twins-FPN           | PCPVT-S  | 512x512   | 80000   | 6.60     | 27.15          | 43.26 | 44.11         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_pcpvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k_20211201_204132-41acd132.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_pcpvt-s_fpn_fpnhead_8x4_512x512_80k_ade20k_20211201_204132.log.json) |
| Twins-UPerNet       | PCPVT-S  | 512x512   | 160000  | 9.67     | 14.24          | 46.04 | 46.92         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_pcpvt-s_uperhead_8xb4-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k/twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k_20211201_233537-8e99c07a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k/twins_pcpvt-s_uperhead_8x4_512x512_160k_ade20k_20211201_233537.log.json)         |
| Twins-FPN           | PCPVT-B  | 512x512   | 80000   | 8.41     | 19.67          | 45.66 | 46.48         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_pcpvt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-b_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_pcpvt-b_fpn_fpnhead_8x4_512x512_80k_ade20k_20211130_141019-d396db72.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-b_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_pcpvt-b_fpn_fpnhead_8x4_512x512_80k_ade20k_20211130_141019.log.json) |
| Twins-UPerNet (8x2) | PCPVT-B  | 512x512   | 160000  | 6.46     | 12.04          | 47.91 | 48.64         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_pcpvt-b_uperhead_8xb2-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-b_uperhead_8x2_512x512_160k_ade20k/twins_pcpvt-b_uperhead_8x2_512x512_160k_ade20k_20211130_141020-02094ea5.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-b_uperhead_8x2_512x512_160k_ade20k/twins_pcpvt-b_uperhead_8x2_512x512_160k_ade20k_20211130_141020.log.json)         |
| Twins-FPN           | PCPVT-L  | 512x512   | 80000   | 10.78    | 14.32          | 45.94 | 46.70         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_pcpvt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-l_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_pcpvt-l_fpn_fpnhead_8x4_512x512_80k_ade20k_20211201_105226-bc6d61dc.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-l_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_pcpvt-l_fpn_fpnhead_8x4_512x512_80k_ade20k_20211201_105226.log.json) |
| Twins-UPerNet (8x2) | PCPVT-L  | 512x512   | 160000  | 7.82     | 10.70          | 49.35 | 50.08         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-l_uperhead_8x2_512x512_160k_ade20k/twins_pcpvt-l_uperhead_8x2_512x512_160k_ade20k_20211201_075053-c6095c07.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_pcpvt-l_uperhead_8x2_512x512_160k_ade20k/twins_pcpvt-l_uperhead_8x2_512x512_160k_ade20k_20211201_075053.log.json)         |
| Twins-FPN           | SVT-S    | 512x512   | 80000   | 5.80     | 29.79          | 44.47 | 45.42         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_svt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-s_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_svt-s_fpn_fpnhead_8x4_512x512_80k_ade20k_20211130_141006-0a0d3317.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-s_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_svt-s_fpn_fpnhead_8x4_512x512_80k_ade20k_20211130_141006.log.json)         |
| Twins-UPerNet (8x2) | SVT-S    | 512x512   | 160000  | 4.93     | 15.09          | 46.08 | 46.96         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_svt-s_uperhead_8xb2-160k_ade20k-512x512.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-s_uperhead_8x2_512x512_160k_ade20k/twins_svt-s_uperhead_8x2_512x512_160k_ade20k_20211130_141005-e48a2d94.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-s_uperhead_8x2_512x512_160k_ade20k/twins_svt-s_uperhead_8x2_512x512_160k_ade20k_20211130_141005.log.json)                 |
| Twins-FPN           | SVT-B    | 512x512   | 80000   | 8.75     | 21.10          | 46.77 | 47.47         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_svt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k_20211201_113849-88b2907c.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k_20211201_113849.log.json)         |
| Twins-UPerNet (8x2) | SVT-B    | 512x512   | 160000  | 6.77     | 12.66          | 48.04 | 48.87         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_svt-b_uperhead_8xb2-160k_ade20k-512x512.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-b_uperhead_8x2_512x512_160k_ade20k/twins_svt-b_uperhead_8x2_512x512_160k_ade20k_20211202_040826-0943a1f1.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-b_uperhead_8x2_512x512_160k_ade20k/twins_svt-b_uperhead_8x2_512x512_160k_ade20k_20211202_040826.log.json)                 |
| Twins-FPN           | SVT-L    | 512x512   | 80000   | 11.20    | 17.80          | 46.55 | 47.74         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_svt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-l_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_svt-l_fpn_fpnhead_8x4_512x512_80k_ade20k_20211130_141005-1d59bee2.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-l_fpn_fpnhead_8x4_512x512_80k_ade20k/twins_svt-l_fpn_fpnhead_8x4_512x512_80k_ade20k_20211130_141005.log.json)         |
| Twins-UPerNet (8x2) | SVT-L    | 512x512   | 160000  | 8.41     | 10.73          | 49.65 | 50.63         | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/twins/twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-l_uperhead_8x2_512x512_160k_ade20k/twins_svt-l_uperhead_8x2_512x512_160k_ade20k_20211130_141005-3e2cae61.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/twins/twins_svt-l_uperhead_8x2_512x512_160k_ade20k/twins_svt-l_uperhead_8x2_512x512_160k_ade20k_20211130_141005.log.json)                 |

Note:

- `8x2` means 8 GPUs with 2 samples per GPU in training. Default setting of Twins on ADE20K is 8 GPUs with 4 samples per GPU in training.
- `UPerNet` and `FPN` are decoder heads utilized in corresponding Twins model, which is `UPerHead` and `FPNHead`, respectively. Specifically, models in [official repo](https://github.com/Meituan-AutoML/Twins) all use `UPerHead`.
