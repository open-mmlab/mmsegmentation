# Vision Transformer for Dense Prediction

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{dosoViTskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={DosoViTskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}

@article{Ranftl2021,
  author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
  title     = {Vision Transformers for Dense Prediction},
  journal   = {ArXiv preprint},
  year      = {2021},
}
```

## How to use ViT pretrained weights

We convert the backbone weights from the pytorch-image-models repository (https://github.com/rwightman/pytorch-image-models) with `tools/model_converters/vit_convert.py`.

You may follow below steps to start DPT training preparation:

1. Download ViT pretrained weights (Suggest put in `pretrain/`);
2. Run convert script to convert official pretrained weights: `python tools/model_converters/vit_convert.py pretrain/vit-timm.pth pretrain/vit-mmseg.pth`;
3. Modify `pretrained` of VisionTransformer model config, for example, `pretrained` of `dpt_vit-b16.py` is set to `pretrain/vit-mmseg.pth`;

## Results and models

### ADE20K

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                               |
| ------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DPT | ViT-B | 512x512  | 160000  | 8.09 | 10.41 | 46.97 | 48.34 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dpt/dpt_vit-b16_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dpt/dpt_vit-b16_512x512_160k_ade20k/dpt_vit-b16_512x512_160k_ade20k-db31cf52.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dpt/dpt_vit-b16_512x512_160k_ade20k/dpt_vit-b16_512x512_160k_ade20k-20210809_172025.log.json) |
