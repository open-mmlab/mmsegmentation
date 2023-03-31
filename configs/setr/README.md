# SETR

> [Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/abs/2012.15840)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/fudan-zvg/SETR">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/setr_up_head.py#L11">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Most recent semantic segmentation methods adopt a fully-convolutional network (FCN) with an encoder-decoder architecture. The encoder progressively reduces the spatial resolution and learns more abstract/semantic visual concepts with larger receptive fields. Since context modeling is critical for segmentation, the latest efforts have been focused on increasing the receptive field, through either dilated/atrous convolutions or inserting attention modules. However, the encoder-decoder based FCN architecture remains unchanged. In this paper, we aim to provide an alternative perspective by treating semantic segmentation as a sequence-to-sequence prediction task. Specifically, we deploy a pure transformer (ie, without convolution and resolution reduction) to encode an image as a sequence of patches. With the global context modeled in every layer of the transformer, this encoder can be combined with a simple decoder to provide a powerful segmentation model, termed SEgmentation TRansformer (SETR). Extensive experiments show that SETR achieves new state of the art on ADE20K (50.28% mIoU), Pascal Context (55.83% mIoU) and competitive results on Cityscapes. Particularly, we achieve the first position in the highly competitive ADE20K test server leaderboard on the day of submission.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142902777-ee2d34b7-a631-4fa7-ad68-118ff5716afe.png" width="80%"/>
</div>

```None
This head has two version head.
```

## Usage

You can download the pretrain from [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth). Then you can convert its keys with the script `vit2mmseg.py` in the tools directory.

```shell
python tools/model_converters/vit2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

E.g.

```shell
python tools/model_converters/vit2mmseg.py \
jx_vit_large_p16_384-b3be5167.pth pretrain/vit_large_p16.pth
```

This script convert the model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20K

| Method     | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | Device | mIoU  | mIoU(ms+flip) | config                                                                                                                     | download                                                                                                                                                                                                                                                                                                                             |
| ---------- | -------- | --------- | ---------- | ------- | -------- | -------------- | ------ | ----- | ------------: | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SETR Naive | ViT-L    | 512x512   | 16         | 160000  | 18.40    | 4.72           | V100   | 48.28 |         49.56 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_512x512_160k_b16_ade20k/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_512x512_160k_b16_ade20k/setr_naive_512x512_160k_b16_ade20k_20210619_191258.log.json) |
| SETR PUP   | ViT-L    | 512x512   | 16         | 160000  | 19.54    | 4.50           | V100   | 48.24 |         49.99 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/setr/setr_vit-l_pup_8xb2-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_512x512_160k_b16_ade20k/setr_pup_512x512_160k_b16_ade20k_20210619_191343-7e0ce826.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_512x512_160k_b16_ade20k/setr_pup_512x512_160k_b16_ade20k_20210619_191343.log.json)         |
| SETR MLA   | ViT-L    | 512x512   | 8          | 160000  | 10.96    | -              | V100   | 47.34 |         49.05 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/setr/setr_vit-l-mla_8xb1-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210619_191118-c6d21df0.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210619_191118.log.json)             |
| SETR MLA   | ViT-L    | 512x512   | 16         | 160000  | 17.30    | 5.25           | V100   | 47.39 |         49.37 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/setr/setr_vit-l_mla_8xb2-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b16_ade20k/setr_mla_512x512_160k_b16_ade20k_20210619_191057-f9741de7.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b16_ade20k/setr_mla_512x512_160k_b16_ade20k_20210619_191057.log.json)         |

### Cityscapes

| Method     | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | Device | mIoU  | mIoU(ms+flip) | config                                                                                                                        | download                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | -------- | --------- | ---------- | ------- | -------- | -------------- | ------ | ----- | ------------: | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SETR Naive | ViT-L    | 768x768   | 8          | 80000   | 24.06    | 0.39           | V100   | 78.10 |         80.22 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/setr/setr_vit-l_naive_8xb1-80k_cityscapes-768x768.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_vit-large_8x1_768x768_80k_cityscapes/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_vit-large_8x1_768x768_80k_cityscapes/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505.log.json) |
| SETR PUP   | ViT-L    | 768x768   | 8          | 80000   | 27.96    | 0.37           | V100   | 79.21 |         81.02 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/setr/setr_vit-l_pup_8xb1-80k_cityscapes-768x768.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_vit-large_8x1_768x768_80k_cityscapes/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_vit-large_8x1_768x768_80k_cityscapes/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115.log.json)         |
| SETR MLA   | ViT-L    | 768x768   | 8          | 80000   | 24.10    | 0.41           | V100   | 77.00 |         79.59 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/setr/setr_vit-l_mla_8xb1-80k_cityscapes-768x768.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_vit-large_8x1_768x768_80k_cityscapes/setr_mla_vit-large_8x1_768x768_80k_cityscapes_20211119_101003-7f8dccbe.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_vit-large_8x1_768x768_80k_cityscapes/setr_mla_vit-large_8x1_768x768_80k_cityscapes_20211119_101003.log.json)         |

## Citation

```bibtex
@article{zheng2020rethinking,
  title={Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers},
  author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jianfeng and Xiang, Tao and Torr, Philip HS and others},
  journal={arXiv preprint arXiv:2012.15840},
  year={2020}
}
```
