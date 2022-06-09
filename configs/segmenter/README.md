# Segmenter

[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/rstrudel/segmenter">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.21.0/mmseg/models/decode_heads/segmenter_mask_head.py#L15">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Image segmentation is often ambiguous at the level of individual image patches and requires contextual information to reach label consensus. In this paper we introduce Segmenter, a transformer model for semantic segmentation. In contrast to convolution-based methods, our approach allows to model global context already at the first layer and throughout the network. We build on the recent Vision Transformer (ViT) and extend it to semantic segmentation. To do so, we rely on the output embeddings corresponding to image patches and obtain class labels from these embeddings with a point-wise linear decoder or a mask transformer decoder. We leverage models pre-trained for image classification and show that we can fine-tune them on moderate sized datasets available for semantic segmentation. The linear decoder allows to obtain excellent results already, but the performance can be further improved by a mask transformer generating class masks. We conduct an extensive ablation study to show the impact of the different parameters, in particular the performance is better for large models and small patch sizes. Segmenter attains excellent results for semantic segmentation. It outperforms the state of the art on both ADE20K and Pascal Context datasets and is competitive on Cityscapes.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/148507554-87eb80bd-02c7-4c31-b102-c6141e231ec8.png" width="70%"/>
</div>

```bibtex
@inproceedings{strudel2021segmenter,
  title={Segmenter: Transformer for semantic segmentation},
  author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7262--7272},
  year={2021}
}
```

## Usage

We have provided pretrained models converted from [ViT-AugReg](https://github.com/rwightman/pytorch-image-models/blob/f55c22bebf9d8afc449d317a723231ef72e0d662/timm/models/vision_transformer.py#L54-L106).

If you want to convert keys on your own to use the pre-trained ViT model from [Segmenter](https://github.com/rstrudel/segmenter), we also provide a script [`vitjax2mmseg.py`](../../tools/model_converters/vitjax2mmseg.py) in the tools directory to convert the key of models from [ViT-AugReg](https://github.com/rwightman/pytorch-image-models/blob/f55c22bebf9d8afc449d317a723231ef72e0d662/timm/models/vision_transformer.py#L54-L106) to MMSegmentation style.

```shell
python tools/model_converters/vitjax2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

E.g.

```shell
python tools/model_converters/vitjax2mmseg.py \
Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz \
pretrain/vit_tiny_p16_384.pth
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

In our default setting, pretrained models and their corresponding [ViT-AugReg](https://github.com/rwightman/pytorch-image-models/blob/f55c22bebf9d8afc449d317a723231ef72e0d662/timm/models/vision_transformer.py#L54-L106) models could be defined below:

| pretrained models     | original models                                                                                                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| vit_tiny_p16_384.pth  | ['vit_tiny_patch16_384'](https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz)   |
| vit_small_p16_384.pth | ['vit_small_patch16_384'](https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz) |
| vit_base_p16_384.pth  | ['vit_base_patch16_384'](https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz)  |
| vit_large_p16_384.pth | ['vit_large_patch16_384'](https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz) |

## Results and models

### ADE20K

| Method           | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config                                                                                                                                 | download                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------- | -------- | --------- | ------- | -------- | -------------- | ----- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Segmenter Mask   | ViT-T_16 | 512x512   | 160000  | 1.21     | 27.98          | 39.99 | 40.83         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706.log.json)         |
| Segmenter Linear | ViT-S_16 | 512x512   | 160000  | 1.78     | 28.07          | 45.75 | 46.82         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segmenter/segmenter_vit-s_linear_8x1_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-s_linear_8x1_512x512_160k_ade20k/segmenter_vit-s_linear_8x1_512x512_160k_ade20k_20220105_151713-39658c46.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-s_linear_8x1_512x512_160k_ade20k/segmenter_vit-s_linear_8x1_512x512_160k_ade20k_20220105_151713.log.json) |
| Segmenter Mask   | ViT-S_16 | 512x512   | 160000  | 2.03     | 24.80          | 46.19 | 47.85         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segmenter/segmenter_vit-s_mask_8x1_512x512_160k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-s_mask_8x1_512x512_160k_ade20k/segmenter_vit-s_mask_8x1_512x512_160k_ade20k_20220105_151706-511bb103.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-s_mask_8x1_512x512_160k_ade20k/segmenter_vit-s_mask_8x1_512x512_160k_ade20k_20220105_151706.log.json)         |
| Segmenter Mask   | ViT-B_16 | 512x512   | 160000  | 4.20     | 13.20          | 49.60 | 51.07         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706.log.json)         |
| Segmenter Mask   | ViT-L_16 | 640x640   | 160000  | 16.56    | 2.62           | 52.16 | 53.65         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segmenter/segmenter_vit-l_mask_8x1_512x512_160k_ade20k.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-l_mask_8x1_512x512_160k_ade20k/segmenter_vit-l_mask_8x1_512x512_160k_ade20k_20220105_162750-7ef345be.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-l_mask_8x1_512x512_160k_ade20k/segmenter_vit-l_mask_8x1_512x512_160k_ade20k_20220105_162750.log.json)         |
