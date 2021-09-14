# Dynamic Multi-scale Filters for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{yu2021bisenet,
  title={Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation},
  author={Yu, Changqian and Gao, Changxin and Wang, Jingbo and Yu, Gang and Shen, Chunhua and Sang, Nong},
  journal={International Journal of Computer Vision},
  pages={1--18},
  year={2021},
  publisher={Springer}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                | download                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BiSeNetV2  | BiSeNetV2  | 1024x1024   |   160000 | 7.64     | 30.88          | 73.21 |         75.74 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-79e63fdc.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551.log.json)         |
| BiSeNetV2 (OHEM)  | BiSeNetV2 | 1024x1024   |   160000 | 7.64       | 33.12          | 73.57 |         75.80 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes_20210902_112947-b31ca269.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes_20210902_112947.log.json)     |
| BiSeNetV2 (FP16)  | BiSeNetV2  | 1024x1024   |  160000 | 5.77         | 35.49              | 73.07 |         75.13 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942-e404ba78.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942.log.json)     |
| BiSeNetV2 (4x8) | BiSeNetV2 | 1024x1024   |  160000 | 15.05        | 31.85              | 75.76 |         77.79 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-ef56feaa.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032.log.json) |
