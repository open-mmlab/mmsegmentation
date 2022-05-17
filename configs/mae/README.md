# MAE

[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/facebookresearch/mae">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.24.0/mmseg/models/backbones/mae.py#L46">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3x or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pre-training and shows promising scaling behavior.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/165456416-1cba54bf-b1b5-4bdf-ad86-d6390de7f342.png" width="70%"/>
</div>

## Citation

```bibtex
@article{he2021masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv preprint arXiv:2111.06377},
  year={2021}
}
```

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`beit2mmseg.py`](../../tools/model_converters/beit2mmseg.py) in the tools directory to convert the key of MAE model from [the official repo](https://github.com/facebookresearch/mae) to MMSegmentation style.

```shell
python tools/model_converters/beit2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

E.g.

```shell
python tools/model_converters/beit2mmseg.py https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth pretrain/mae_pretrain_vit_base_mmcls.pth
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

In our default setting, pretrained models could be defined below:

| pretrained models               | original models                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| mae_pretrain_vit_base_mmcls.pth | ['mae_pretrain_vit_base'](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) |

Verify the single-scale results of the model:

```shell
sh tools/dist_test.sh \
configs/mae/upernet_mae-base_fp16_8x2_512x512_160k_ade20k.py \
upernet_mae-base_fp16_8x2_512x512_160k_ade20k_20220426_174752-f92a2975.pth $GPUS --eval mIoU
```

Since relative position embedding requires the input length and width to be equal, the sliding window is adopted for multi-scale inference. So we set min_size=512, that is, the shortest edge is 512. So the multi-scale inference of config is performed separately, instead of '--aug-test'. For multi-scale inference:

```shell
sh tools/dist_test.sh \
configs/mae/upernet_mae-base_fp16_512x512_160k_ade20k_ms.py \
upernet_mae-base_fp16_8x2_512x512_160k_ade20k_20220426_174752-f92a2975.pth $GPUS --eval mIoU
```

## Results and models

### ADE20K

| Method  | Backbone | Crop Size | pretrain    | pretrain img size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                                       |
| ------- | -------- | --------- | ----------- | ----------------- | ---------- | ------- | -------- | -------------- | ----- | ------------: | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UperNet | ViT-B    | 512x512   | ImageNet-1K | 224x224           | 16         | 160000  | 9.96     | 7.14           | 48.13 |         48.70 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mae/upernet_mae-base_fp16_8x2_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mae/upernet_mae-base_fp16_8x2_512x512_160k_ade20k/upernet_mae-base_fp16_8x2_512x512_160k_ade20k_20220426_174752-f92a2975.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/mae/upernet_mae-base_fp16_8x2_512x512_160k_ade20k/upernet_mae-base_fp16_8x2_512x512_160k_ade20k_20220426_174752.log.json) |
