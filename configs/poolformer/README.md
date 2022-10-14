# PoolFormer

[MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/sail-sg/poolformer/tree/main/segmentation">Official Repo</a>

<a href="https://github.com/open-mmlab/mmclassification/blob/v0.23.0/mmcls/models/backbones/poolformer.py#L198">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model's performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only the most basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 48%/60% fewer MACs. The effectiveness of PoolFormer verifies our hypothesis and urges us to initiate the concept of "MetaFormer", a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design. Code is available at [this https URL](https://github.com/sail-sg/poolformer)

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15921929/144710761-1635f59a-abde-4946-984c-a2c3f22a19d2.png" width="70%"/>
</div>

## Citation

```bibtex
@inproceedings{yu2022metaformer,
  title={Metaformer is actually what you need for vision},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10819--10829},
  year={2022}
}
```

### Usage

- PoolFormer backbone needs to install [MMClassification](https://github.com/open-mmlab/mmclassification) first, which has abundant backbones for downstream tasks.

```shell
pip install "mmcls>=1.0.0rc0"
```

- The pretrained models could also be downloaded from [PoolFormer config of MMClassification](https://github.com/open-mmlab/mmclassification/tree/master/configs/poolformer).

## Results and models

### ADE20K

| Method | Backbone       | Crop Size | pretrain    | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | mIoU\* | mIoU\*(ms+flip) | config                                                                                                                               | download                                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------------- | --------- | ----------- | ---------- | ------- | -------- | -------------- | ----- | ------------: | ------ | --------------: | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FPN    | PoolFormer-S12 | 512x512   | ImageNet-1K | 32         | 40000   | 4.17     | 23.48          | 36.68 |             - | 37.07  |               - | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/poolformer/fpn_poolformer_s12_8xb4-40k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_s12_8x4_512x512_40k_ade20k/fpn_poolformer_s12_8x4_512x512_40k_ade20k_20220501_115154-b5aa2f49.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_s12_8x4_512x512_40k_ade20k/fpn_poolformer_s12_8x4_512x512_40k_ade20k_20220501_115154.log.json) |
| FPN    | PoolFormer-S24 | 512x512   | ImageNet-1K | 32         | 40000   | 5.47     | 15.74          | 40.12 |             - | 40.36  |               - | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/poolformer/fpn_poolformer_s24_8xb4-40k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_s24_8x4_512x512_40k_ade20k/fpn_poolformer_s24_8x4_512x512_40k_ade20k_20220503_222049-394a7cf7.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_s24_8x4_512x512_40k_ade20k/fpn_poolformer_s24_8x4_512x512_40k_ade20k_20220503_222049.log.json) |
| FPN    | PoolFormer-S36 | 512x512   | ImageNet-1K | 32         | 40000   | 6.77     | 11.34          | 41.61 |             - | 41.81  |               - | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/poolformer/fpn_poolformer_s36_8xb4-40k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_s36_8x4_512x512_40k_ade20k/fpn_poolformer_s36_8x4_512x512_40k_ade20k_20220501_151122-b47e607d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_s36_8x4_512x512_40k_ade20k/fpn_poolformer_s36_8x4_512x512_40k_ade20k_20220501_151122.log.json) |
| FPN    | PoolFormer-M36 | 512x512   | ImageNet-1K | 32         | 40000   | 8.59     | 8.97           | 41.95 |             - | 42.35  |               - | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/poolformer/fpn_poolformer_m36_8xb4-40k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_m36_8x4_512x512_40k_ade20k/fpn_poolformer_m36_8x4_512x512_40k_ade20k_20220501_164230-3dc83921.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_m36_8x4_512x512_40k_ade20k/fpn_poolformer_m36_8x4_512x512_40k_ade20k_20220501_164230.log.json) |
| FPN    | PoolFormer-M48 | 512x512   | ImageNet-1K | 32         | 40000   | 10.48    | 6.69           | 42.43 |             - | 42.76  |               - | [config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/poolformer/fpn_poolformer_m48_8xb4-40k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_m48_8x4_512x512_40k_ade20k/fpn_poolformer_m48_8x4_512x512_40k_ade20k_20220504_003923-64168d3b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/fpn_poolformer_m48_8x4_512x512_40k_ade20k/fpn_poolformer_m48_8x4_512x512_40k_ade20k_20220504_003923.log.json) |

Note:

- We replace `AlignedResize` in original PoolFormer implementation to `Resize + ResizeToMultiple`.

- `mIoU` with * is collected when `Resize + ResizeToMultiple` is adopted in `test_pipeline`, so do `mIoU` in logs.

- The Test Time Augmentation i.e., "ms+flip" in MMSegmentation v1.x is developing, stay tuned!
