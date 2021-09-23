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

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`vit2mmseg.py`](../../tools/model_converters/vit2mmseg.py) in the tools directory to convert the key of models from [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) to MMSegmentation style.

```shell
python tools/model_converters/vit2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

E.g.

```shell
python tools/model_converters/vit2mmseg.py https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth pretrain/jx_vit_base_p16_224-80ecf9dd.pth
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20K

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                               |
| ------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DPT | ViT-B | 512x512  | 160000  | 8.09 | 10.41 | 46.97 | 48.34 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dpt/dpt_vit-b16_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/dpt/dpt_vit-b16_512x512_160k_ade20k/dpt_vit-b16_512x512_160k_ade20k-db31cf52.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/dpt/dpt_vit-b16_512x512_160k_ade20k/dpt_vit-b16_512x512_160k_ade20k-20210809_172025.log.json) |
