# Dynamic Multi-scale Filters for Semantic Segmentation

## Introduction

[ALGORITHM]

```latex
@InProceedings{He_2019_ICCV,
author = {He, Junjun and Deng, Zhongying and Qiao, Yu},
title = {Dynamic Multi-Scale Filters for Semantic Segmentation},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                                 download                                                                                                                                                                                                 |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DMNet | R-50-D8  | 512x1024  |   40000 |      7.0 |           3.66 | 77.78 |         79.14 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_40k_cityscapes/dmnet_r50-d8_512x1024_40k_cityscapes_20201214_115717-5e88fa33.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_40k_cityscapes/dmnet_r50-d8_512x1024_40k_cityscapes-20201214_115717.log.json)     |
| DMNet | R-101-D8 | 512x1024  |   40000 |      10.6 |           2.54 | 78.37 |         79.72 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x1024_40k_cityscapes/dmnet_r101-d8_512x1024_40k_cityscapes_20201214_115716-abc9d111.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x1024_40k_cityscapes/dmnet_r101-d8_512x1024_40k_cityscapes-20201214_115716.log.json) |
| DMNet | R-50-D8  | 769x769   |   40000 |      7.9 |           1.57 | 78.49 |         80.27 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_769x769_40k_cityscapes/dmnet_r50-d8_769x769_40k_cityscapes_20201214_115717-2a2628d7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_769x769_40k_cityscapes/dmnet_r50-d8_769x769_40k_cityscapes-20201214_115717.log.json)         |
| DMNet | R-101-D8 | 769x769   |   40000 |     12.0 |           1.01 | 77.62 |         78.94 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_769x769_40k_cityscapes/dmnet_r101-d8_769x769_40k_cityscapes_20201214_115718-b650de90.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_769x769_40k_cityscapes/dmnet_r101-d8_769x769_40k_cityscapes-20201214_115718.log.json)     |
| DMNet | R-50-D8  | 512x1024  |   80000 | -        | -              | 79.07 |         80.22 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_80k_cityscapes/dmnet_r50-d8_512x1024_80k_cityscapes_20201214_115716-987f51e3.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x1024_80k_cityscapes/dmnet_r50-d8_512x1024_80k_cityscapes-20201214_115716.log.json)     |
| DMNet | R-101-D8 | 512x1024  |   80000 | -        | -              | 79.64 |         80.67 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x1024_80k_cityscapes/dmnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x1024_80k_cityscapes/dmnet_r101-d8_512x1024_80k_cityscapes-20201214_115705.log.json) |
| DMNet | R-50-D8  | 769x769   |   80000 | -        | -              | 79.22 |         80.55 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_769x769_80k_cityscapes/dmnet_r50-d8_769x769_80k_cityscapes_20201214_115718-7ea9fa12.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_769x769_80k_cityscapes/dmnet_r50-d8_769x769_80k_cityscapes-20201214_115718.log.json)         |
| DMNet | R-101-D8 | 769x769   |   80000 | -        | -              | 79.19 |         80.65 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_769x769_80k_cityscapes/dmnet_r101-d8_769x769_80k_cityscapes_20201214_115716-a7fbc2ab.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_769x769_80k_cityscapes/dmnet_r101-d8_769x769_80k_cityscapes-20201214_115716.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) |                                                                                                                                                                                         download                                                                                                                                                                                         |
|--------|----------|-----------|--------:|----------|----------------|------:|--------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DMNet | R-50-D8  | 512x512   |   80000 |      9.4 |          20.95 | 42.37 |         43.62 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x512_80k_ade20k/dmnet_r50-d8_512x512_80k_ade20k_20201214_115705-a8626293.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x512_80k_ade20k/dmnet_r50-d8_512x512_80k_ade20k-20201214_115705.log.json)         |
| DMNet | R-101-D8 | 512x512   |   80000 |       13.0 |          13.88 | 45.34 |         46.13 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x512_80k_ade20k/dmnet_r101-d8_512x512_80k_ade20k_20201214_115704-c656c3fb.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x512_80k_ade20k/dmnet_r101-d8_512x512_80k_ade20k-20201214_115704.log.json)     |
| DMNet | R-50-D8  | 512x512   |  160000 | -        | -              | 43.15 |         44.17 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x512_160k_ade20k/dmnet_r50-d8_512x512_160k_ade20k_20201214_115706-25fb92c2.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r50-d8_512x512_160k_ade20k/dmnet_r50-d8_512x512_160k_ade20k-20201214_115706.log.json)     |
| DMNet | R-101-D8 | 512x512   |  160000 | -        | -              | 45.42 |         46.76 | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x512_160k_ade20k/dmnet_r101-d8_512x512_160k_ade20k_20201214_115705-73f9a8d7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dmnet/dmnet_r101-d8_512x512_160k_ade20k/dmnet_r101-d8_512x512_160k_ade20k-20201214_115705.log.json) |
