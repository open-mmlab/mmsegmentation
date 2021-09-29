# Bisenet v2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/backbones/bisenetv2.py#L545">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/2004.02147">BiSeNetV2 (IJCV'2021)</a></summary>

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

</details>

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                | download                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BiSeNetV2  | BiSeNetV2  | 1024x1024   |   160000 | 7.64     | 31.77         | 73.21 |         75.74 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551.log.json)         |
| BiSeNetV2 (OHEM)  | BiSeNetV2 | 1024x1024   |   160000 | 7.64       | -          | 73.57 |         75.80 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes_20210902_112947-5f8103b4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_ohem_4x4_1024x1024_160k_cityscapes_20210902_112947.log.json)     |
| BiSeNetV2 (4x8) | BiSeNetV2 | 1024x1024   |  160000 | 15.05        | -              | 75.76 |         77.79 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032.log.json) |
| BiSeNetV2 (FP16)  | BiSeNetV2  | 1024x1024   |  160000 | 5.77         | 36.65              | 73.07 |         75.13 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942-b979777b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes_20210902_045942.log.json)     |

Note:

- `OHEM` means Online Hard Example Mining (OHEM) is adopted in training.
- `FP16` means Mixed Precision (FP16) is adopted in training.
- `4x8` means 4 GPUs with 8 samples per GPU in training.
