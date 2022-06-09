# FastFCN

[FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation](https://arxiv.org/abs/1903.11816)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/wuhuikai/FastFCN">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/necks/jpu.py#L12">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Modern approaches for semantic segmentation usually employ dilated convolutions in the backbone to extract high-resolution feature maps, which brings heavy computation complexity and memory footprint. To replace the time and memory consuming dilated convolutions, we propose a novel joint upsampling module named Joint Pyramid Upsampling (JPU) by formulating the task of extracting high-resolution feature maps into a joint upsampling problem. With the proposed JPU, our method reduces the computation complexity by more than three times without performance loss. Experiments show that JPU is superior to other upsampling modules, which can be plugged into many existing approaches to reduce computation complexity and improve performance. By replacing dilated convolutions with the proposed JPU module, our method achieves the state-of-the-art performance in Pascal Context dataset (mIoU of 53.13%) and ADE20K dataset (final score of 0.5584) while running 3 times faster.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142901374-6e0252ab-6e0f-4acd-86ad-1e9f49be3185.png" width="70%"/>
</div>

## Citation

```bibtex
@article{wu2019fastfcn,
title={Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation},
author={Wu, Huikai and Zhang, Junge and Huang, Kaiqi and Liang, Kongming and Yu, Yizhou},
journal={arXiv preprint arXiv:1903.11816},
year={2019}
}
```

## Results and models

### Cityscapes

| Method                    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FastFCN + DeepLabV3       | R-50-D32 | 512x1024  |   80000 | 5.67     | 2.64           | 79.12 | 80.58         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes_20210928_053722-5d1a2648.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes_20210928_053722.log.json)                 |
| FastFCN + DeepLabV3 (4x4) | R-50-D32 | 512x1024  |   80000 | 9.79     | -              | 79.52 | 80.91         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_4x4_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_aspp_4x4_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_aspp_4x4_512x1024_80k_cityscapes_20210924_214357-72220849.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_aspp_4x4_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_aspp_4x4_512x1024_80k_cityscapes_20210924_214357.log.json) |
| FastFCN + PSPNet          | R-50-D32 | 512x1024  |   80000 | 5.67     | 4.40           | 79.26 | 80.86         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_psp_512x1024_80k_cityscapes.py)      | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_psp_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_psp_512x1024_80k_cityscapes_20210928_053722-57749bed.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_psp_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_psp_512x1024_80k_cityscapes_20210928_053722.log.json)                     |
| FastFCN + PSPNet (4x4)    | R-50-D32 | 512x1024  |   80000 | 9.94     | -              | 78.76 | 80.03         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_psp_4x4_512x1024_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_psp_4x4_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_psp_4x4_512x1024_80k_cityscapes_20210925_061841-77e87b0a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_psp_4x4_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_psp_4x4_512x1024_80k_cityscapes_20210925_061841.log.json)     |
| FastFCN + EncNet          | R-50-D32 | 512x1024  |   80000 | 8.15     | 4.77           | 77.97 | 79.92         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_enc_512x1024_80k_cityscapes.py)      | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_enc_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_enc_512x1024_80k_cityscapes_20210928_030036-78da5046.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_enc_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_enc_512x1024_80k_cityscapes_20210928_030036.log.json)                     |
| FastFCN + EncNet (4x4)    | R-50-D32 | 512x1024  |   80000 | 15.45    | -              |  78.6 | 80.25         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_enc_4x4_512x1024_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_enc_4x4_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_enc_4x4_512x1024_80k_cityscapes_20210926_093217-e1eb6dbb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_enc_4x4_512x1024_80k_cityscapes/fastfcn_r50-d32_jpu_enc_4x4_512x1024_80k_cityscapes_20210926_093217.log.json)     |

### ADE20K

| Method              | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                             | download                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FastFCN + DeepLabV3 | R-50-D32 | 512x512   |   80000 | 8.46     | 12.06          | 41.88 | 42.91         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_512x512_80k_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_aspp_512x512_80k_ade20k/fastfcn_r50-d32_jpu_aspp_512x512_80k_ade20k_20211013_190619-3aa40f2d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_aspp_512x512_80k_ade20k/fastfcn_r50-d32_jpu_aspp_512x512_80k_ade20k_20211013_190619.log.json)     |
| FastFCN + DeepLabV3 | R-50-D32 | 512x512   |  160000 | -        | -              | 43.58 | 44.92         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_aspp_512x512_160k_ade20k/fastfcn_r50-d32_jpu_aspp_512x512_160k_ade20k_20211008_152246-27036aee.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_aspp_512x512_160k_ade20k/fastfcn_r50-d32_jpu_aspp_512x512_160k_ade20k_20211008_152246.log.json) |
| FastFCN + PSPNet    | R-50-D32 | 512x512   |   80000 | 8.02     | 19.21          | 41.40 | 42.12         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_psp_512x512_80k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_psp_512x512_80k_ade20k/fastfcn_r50-d32_jpu_psp_512x512_80k_ade20k_20210930_225137-993d07c8.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_psp_512x512_80k_ade20k/fastfcn_r50-d32_jpu_psp_512x512_80k_ade20k_20210930_225137.log.json)         |
| FastFCN + PSPNet    | R-50-D32 | 512x512   |  160000 | -        | -              | 42.63 | 43.71         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_psp_512x512_160k_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_psp_512x512_160k_ade20k/fastfcn_r50-d32_jpu_psp_512x512_160k_ade20k_20211008_105455-e8f5a2fd.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_psp_512x512_160k_ade20k/fastfcn_r50-d32_jpu_psp_512x512_160k_ade20k_20211008_105455.log.json)     |
| FastFCN + EncNet    | R-50-D32 | 512x512   |   80000 | 9.67     | 17.23          | 40.88 | 42.36         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_enc_512x512_80k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_enc_512x512_80k_ade20k/fastfcn_r50-d32_jpu_enc_512x512_80k_ade20k_20210930_225214-65aef6dd.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_enc_512x512_80k_ade20k/fastfcn_r50-d32_jpu_enc_512x512_80k_ade20k_20210930_225214.log.json)         |
| FastFCN + EncNet    | R-50-D32 | 512x512   |  160000 | -        | -              | 42.50 | 44.21         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn_r50-d32_jpu_enc_512x512_160k_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_enc_512x512_160k_ade20k/fastfcn_r50-d32_jpu_enc_512x512_160k_ade20k_20211008_105456-d875ce3c.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn_r50-d32_jpu_enc_512x512_160k_ade20k/fastfcn_r50-d32_jpu_enc_512x512_160k_ade20k_20211008_105456.log.json)     |

Note:

- `4x4` means 4 GPUs with 4 samples per GPU in training, default setting is 4 GPUs with 2 samples per GPU in training.
- Results of [DeepLabV3 (mIoU: 79.32)](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3), [PSPNet (mIoU: 78.55)](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/pspnet) and [ENCNet (mIoU: 77.94)](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/encnet) can be found in each original repository.
