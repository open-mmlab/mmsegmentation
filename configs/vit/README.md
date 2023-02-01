# Vision Transformer

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/google-research/vision_transformer">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/vit.py#L98">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142903144-f80a12cc-8698-48ab-843c-49dedf558121.png" width="70%"/>
</div>

## Citation

```bibtex
@article{dosoViTskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={DosoViTskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
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

| Method  | Backbone          | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                     | download                                                                                                                                                                                                                                                                                                                   |
| ------- | ----------------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UPerNet | ViT-B + MLN       | 512x512   |   80000 | 9.20     | 6.94           | 47.71 |         49.51 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py)         | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_80k_ade20k/upernet_vit-b16_mln_512x512_80k_ade20k_20210624_130547-0403cee1.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_80k_ade20k/20210624_130547.log.json)                |
| UPerNet | ViT-B + MLN       | 512x512   |  160000 | 9.20     | 7.58           | 46.75 |         48.46 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_160k_ade20k/upernet_vit-b16_mln_512x512_160k_ade20k_20210624_130547-852fa768.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_160k_ade20k/20210623_192432.log.json)             |
| UPerNet | ViT-B + LN + MLN  | 512x512   |  160000 | 9.21     | 6.82           | 47.73 |         49.95 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k/upernet_vit-b16_ln_mln_512x512_160k_ade20k_20210621_172828-f444c077.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k/20210621_172828.log.json)    |
| UPerNet | DeiT-S            | 512x512   |   80000 | 4.68     | 29.85          | 42.96 |         43.79 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_deit-s16_upernet_8xb2-80k_ade20k-512x512.py)            | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_80k_ade20k/upernet_deit-s16_512x512_80k_ade20k_20210624_095228-afc93ec2.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_80k_ade20k/20210624_095228.log.json)                         |
| UPerNet | DeiT-S            | 512x512   |  160000 | 4.68     | 29.19          | 42.87 |         43.79 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_deit-s16_upernet_8xb2-160k_ade20k-512x512.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_160k_ade20k/upernet_deit-s16_512x512_160k_ade20k_20210621_160903-5110d916.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_512x512_160k_ade20k/20210621_160903.log.json)                      |
| UPerNet | DeiT-S + MLN      | 512x512   |  160000 | 5.69     | 11.18          | 43.82 |         45.07 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512.py)       | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_mln_512x512_160k_ade20k/upernet_deit-s16_mln_512x512_160k_ade20k_20210621_161021-fb9a5dfb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_mln_512x512_160k_ade20k/20210621_161021.log.json)          |
| UPerNet | DeiT-S + LN + MLN | 512x512   |  160000 | 5.69     | 12.39          | 43.52 |         45.01 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_deit-s16-ln_mln_upernet_512x512_160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k/upernet_deit-s16_ln_mln_512x512_160k_ade20k_20210621_161021-c0cd652f.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k/20210621_161021.log.json) |
| UPerNet | DeiT-B            | 512x512   |   80000 | 7.75     | 9.69           | 45.24 |         46.73 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_deit-b16_upernet_8xb2-80k_ade20k-512x512.py)            | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_80k_ade20k/upernet_deit-b16_512x512_80k_ade20k_20210624_130529-1e090789.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_80k_ade20k/20210624_130529.log.json)                         |
| UPerNet | DeiT-B            | 512x512   |  160000 | 7.75     | 10.39          | 45.36 |         47.16 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_deit-b16_upernet_8xb2-160k_ade20k-512x512.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_160k_ade20k/upernet_deit-b16_512x512_160k_ade20k_20210621_180100-828705d7.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_512x512_160k_ade20k/20210621_180100.log.json)                      |
| UPerNet | DeiT-B + MLN      | 512x512   |  160000 | 9.21     | 7.78           | 45.46 |         47.16 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_deit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py)       | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_mln_512x512_160k_ade20k/upernet_deit-b16_mln_512x512_160k_ade20k_20210621_191949-4e1450f3.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_mln_512x512_160k_ade20k/20210621_191949.log.json)          |
| UPerNet | DeiT-B + LN + MLN | 512x512   |  160000 | 9.21     | 7.75           | 45.37 |         47.23 | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/vit/vit_deit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py)    | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_ln_mln_512x512_160k_ade20k/upernet_deit-b16_ln_mln_512x512_160k_ade20k_20210623_153535-8a959c14.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_ln_mln_512x512_160k_ade20k/20210623_153535.log.json) |
