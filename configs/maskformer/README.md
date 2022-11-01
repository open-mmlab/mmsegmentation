# MaskFormer

[MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/facebookresearch/MaskFormer/">Official Repo</a>

<a href="https://github.com/open-mmlab/mmdetection/blob/dev-3.x/mmdet/models/dense_heads/maskformer_head.py#L21">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a single global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/199215459-ea507126-aafe-4823-8eb1-ae6487509d5c.png" width="90%"/>
</div>

```bibtex
@article{cheng2021per,
  title={Per-pixel classification is not all you need for semantic segmentation},
  author={Cheng, Bowen and Schwing, Alex and Kirillov, Alexander},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={17864--17875},
  year={2021}
}
```

## Results and models

### ADE20K

| Method     | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config                                                                                                                                       | download                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------- | --------- | --------- | ------- | -------- | -------------- | ----- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MaskFormer | R-50-D32  | 512x512   | 160000  | 3.29     | 42.20          | 44.29 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/maskformer/maskformer_r50-d32_8xb2-160k_ade20k-512x512.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_043751-abcab920.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_043751.log.json)                         |
| MaskFormer | R-101-D32 | 512x512   | 160000  | 4.12     | 34.90          | 45.11 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/maskformer/maskformer_r101-d32_8xb2-160k_ade20k-512x512.py)       | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_054634-d2c72240.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_054634.log.json)             |
| MaskFormer | Swin-T    | 512x512   | 160000  | 3.73     | 40.53          | 46.72 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/maskformer/maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_041642-00c8fbeb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_041642.log.json) |
| MaskFormer | Swin-S    | 512x512   | 160000  | 999      | 999            | 999   | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/maskformer/maskformer_swin-s_upernet_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k_20220304_125657-215753b0.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k_20220304_125657.log.json)         |

Note:

- All experiments of MaskFormer are implemented with 8 V100 (32G) GPUs with 2 samplers per GPU.
- The ResNet backbones utilized in MaskFormer models are standard `ResNet` rather than `ResNetV1c`.
