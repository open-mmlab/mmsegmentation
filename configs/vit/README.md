# Vision Transformer

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{dosoViTskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={DosoViTskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

## Results and models

### ADE20K

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                               |
| ------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UPerNet | ViT-B     | 512x512  |   80000 |   8.8    |     7.86       |45.99  |  48.06  |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/upernet_vit-b16_512x512_80k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_512x512_80k_ade20k/upernet_vit-b16_512x512_80k_ade20k-d6b6fbb3.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_512x512_80k_ade20k/20210509_175430.log.json)  |
| UPerNet | ViT-B     | 512x512  |   160000 |       |    8.41    |45.88  |  47.9  |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/upernet_vit-b16_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_512x512_160k_ade20k/upernet_vit-b16_512x512_160k_ade20k-178101c0.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_512x512_160k_ade20k/20210512_130043.log.json)  |
| UPerNet | DeiT-S  | 512x512  |   80000 |   5.3    |     14.01      | 41.32 |  42.48  |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/upernet_deit-s16_512x512_80k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_80k_ade20k/upernet_deit-s16_512x512_80k_ade20k-9855ed8a.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_80k_ade20k/20210517_114414.log.json)  |
| UPerNet | DeiT-S  | 512x512  |   160000 |       |    15.05  | 40.61 |  42.04  |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/upernet_deit-s16_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_160k_ade20k/upernet_deit-s16_512x512_160k_ade20k-f96d1a2f.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_160k_ade20k/20210517_114547.log.json)  |
| UPerNet | DeiT-B  | 512x512  |   80000 |  8.9   |      8.51     | 43.31  | 44.95 |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/upernet_deit-b16_512x512_80k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_80k_ade20k/upernet_deit-b16_512x512_80k_ade20k-eb6741cc.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_80k_ade20k/20210518_162229.log.json)  |
| UPerNet | DeiT-B  | 512x512  |   160000 |       |    7.79    | 43.21 | 44.84 |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/upernet_deit-b16_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_160k_ade20k/upernet_deit-b16_512x512_160k_ade20k-3a601a75.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_160k_ade20k/20210519_163905.log.json)  |
