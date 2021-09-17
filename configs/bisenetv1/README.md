# BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{yu2018bisenet,
  title={Bisenet: Bilateral segmentation network for real-time semantic segmentation},
  author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={325--341},
  year={2018}
}
```

## Results and models

### Cityscapes

| Method    | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                  | download                                                                                                                                                                                                                                                       |
| --------- | --------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BiSeNetV1 (4x4) | R-18-D32 | 1024x1024  | 160000 | 3.3 | 31.77 | 74.37 | 76.91 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853.log.json) |
| BiSeNetV1 (4x8) | R-18-D32 | 1024x1024  | 160000 | 3.3 | - | 75.16 | 77.24 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_4x8_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853.log.json) |

Note:

- `4x4`: Using 4 GPUs with 4 samples per GPU in training.
- `4x8`: Using 4 GPUs with 8 samples per GPU in training.
