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
| SETR-Naive | ViT-L | 512x512  | 16          | 160000   | 18.40        | 4.72              | 48.28 |             49.56 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/setr_naive_512x512_160k_b16_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_512x512_160k_b16_ade20k/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_512x512_160k_b16_ade20k/setr_naive_512x512_160k_b16_ade20k_20210619_191258.log.json)     |
| SETR-PUP | ViT-L | 512x512  | 16          | 160000   | 19.54        | 4.50              | 48.24 |             49.99 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/setr_pup_512x512_160k_b16_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_512x512_160k_b16_ade20k/setr_pup_512x512_160k_b16_ade20k_20210619_191343-7e0ce826.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_512x512_160k_b16_ade20k/setr_pup_512x512_160k_b16_ade20k_20210619_191343.log.json)     |
| SETR-MLA | ViT-L | 512x512  | 8           | 160000   | 10.96        | -              | 47.34 |             49.05 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/setr_mla_512x512_160k_b8_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210619_191118-c6d21df0.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210619_191118.log.json)     |
| SETR-MLA | ViT-L | 512x512  | 16          | 160000   | 17.30        | 5.25              | 47.54 |             49.37 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/setr_mla_512x512_160k_b16_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b16_ade20k/setr_mla_512x512_160k_b16_ade20k_20210619_191057-f9741de7.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b16_ade20k/setr_mla_512x512_160k_b16_ade20k_20210619_191057.log.json)     |
