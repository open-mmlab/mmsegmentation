# FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/wuhuikai/FastFCN">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/necks/jpu.py#L12">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/1903.11816">FastFCN (ArXiv'2019) </a></summary>

```latex
@article{wu2019fastfcn,
title={Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation},
author={Wu, Huikai and Zhang, Junge and Huang, Kaiqi and Liang, Kongming and Yu, Yizhou},
journal={arXiv preprint arXiv:1903.11816},
year={2019}
}
```

</details>

## Results and models

### Cityscapes

| Method    | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                  | download                                                                                                                                                                                                                                                       |
| --------- | --------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [DeepLabV3](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3) | R-50-D8         | 512x1024  |   80000 | 6.1        | 2.57              | 79.32 |         80.57 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes.py)         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes/deeplabv3_r50-d8_512x1024_80k_cityscapes_20200606_113404-b92cfdd4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes/deeplabv3_r50-d8_512x1024_80k_cityscapes_20200606_113404.log.json) |
| DeepLabV3 + JPU (4x2) | R-50-D32 | 512x1024  | 80000 | 999 | 999 | 0 | 0 | 0 |0 |
| DeepLabV3 + JPU (4x4) | R-50-D32 | 512x1024  | 80000 | 9.79 | 2.64 | 79.52 | 80.91 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn-jpu_deeplabv3_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn-jpu_deeplabv3_512x1024_80k_cityscapes/fastfcn-jpu_deeplabv3_512x1024_80k_cityscapes_20210924_214357-c2b06737.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn-jpu_deeplabv3_512x1024_80k_cityscapes/fastfcn-jpu_deeplabv3_512x1024_80k_cityscapes_20210924_214357.log.json) |
| [PSPNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/pspnet) | R-50-D8   | 512x1024  |   80000 | 6.1        | 4.07              | 78.55 |         79.79 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes/pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes/pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131.log.json)         |
| PSPNet + JPU (4x2) | R-50-D32   | 512x1024  |   80000 |  5.67       |  4.40             | 79.26 |   80.86       |   |        |
| PSPNet + JPU (4x2) | R-50-D32   | 512x1024  |   80000 |  9.94       |  -             | 78.76 |   80.03       | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn-jpu_psp_512x1024_80k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn-jpu_psp_512x1024_80k_cityscapes/fastfcn-jpu_psp_512x1024_80k_cityscapes_20210925_061841-b07c5e32.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn-jpu_psp_512x1024_80k_cityscapes/fastfcn-jpu_psp_512x1024_80k_cityscapes_20210925_061841.log.json)         |
| [encnet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/encnet) | R-50-D8  | 512x1024  |   80000 | 8.6        | 4.58              | 77.94 |         79.13 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/encnet/encnet_r50-d8_512x1024_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_80k_cityscapes/encnet_r50-d8_512x1024_80k_cityscapes_20200622_003554-fc5c5624.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/encnet/encnet_r50-d8_512x1024_80k_cityscapes/encnet_r50-d8_512x1024_80k_cityscapes-20200622_003554.log.json)     |
| encnet + JPU (4x2)| R-50-D32  | 512x1024  |   80000 | 8.15        |  4.77             | 77.97 |79.92          |   |    |
| encnet + JPU (4x4)| R-50-D32  | 512x1024  |   80000 | 15.45        | -              | 78.6 |         80.25 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastfcn/fastfcn-jpu_enc_512x1024_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn-jpu_enc_512x1024_80k_cityscapes/fastfcn-jpu_enc_512x1024_80k_cityscapes_20210926_093217-c2e5d0fd.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fastfcn/fastfcn-jpu_enc_512x1024_80k_cityscapes/fastfcn-jpu_enc_512x1024_80k_cityscapes_20210926_093217.log.json)     |
