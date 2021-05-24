# Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{zheng2020rethinking,
  title={Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers},
  author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jianfeng and Xiang, Tao and Torr, Philip HS and others},
  journal={arXiv preprint arXiv:2012.15840},
  year={2020}
}
```

## Results and models

### ADE20K

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ---------- | ------- | -------- | -------------- | ----- | ------------: | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SETR-Naive | ViT | 512x512  | 16          | 160000   | -        | -              | 48.47 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/setr_naive_512x512_160k_b16_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_512x512_160k_b16_ade20k/setr_naive_512x512_160k_b16_ade20k_20210521_155347-****.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_512x512_160k_b16_ade20k/setr_naive_512x512_160k_b16_ade20k_20210521_155347.log.json)     |
| SETR-PUP | ViT | 512x512  | 16          | 160000   | -        | -              | 48.59 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/setr_pup_512x512_160k_b16_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_512x512_160k_b16_ade20k/setr_pup_512x512_160k_b16_ade20k_20210521_155425-****.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_512x512_160k_b16_ade20k/setr_pup_512x512_160k_b16_ade20k_20210521_155425.log.json)     |
| SETR-MLA | ViT | 512x512  | 8           | 160000   | -        | -              | 47.63 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/setr_mla_512x512_160k_b8_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210521_160027-******.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210521_160027.log.json)     |
| SETR-MLA | ViT | 512x512  | 16          | 160000   | -        | -              | 47.54 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/setr_mla_512x512_160k_b16_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b16_ade20k/setr_mla_512x512_160k_b16_ade20k_20210521_160043-*****.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b16_ade20k/setr_mla_512x512_160k_b16_ade20k_20210521_160043.log.json)     |
