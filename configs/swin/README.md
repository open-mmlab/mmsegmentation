# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/microsoft/Swin-Transformer">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/swin.py#L524">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/2103.14030">Swin Transformer (arXiv'2021)</a></summary>

```latex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

</details>

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`swin2mmseg.py`](../../tools/model_converters/swin2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) to MMSegmentation style.

```shell
python tools/model_converters/swin2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

E.g.

```shell
python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth pretrain/swin_base_patch4_window7_224.pth
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20K

| Method | Backbone | Crop Size | pretrain | pretrain img size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ---------- | ------- | -------- | --- | --- | -------------- | ----- | ------------: | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UperNet | Swin-T | 512x512 | ImageNet-1K | 224x224 | 16          | 160000   | 5.02        | 21.06              | 44.41 | 45.79            | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542.log.json)     |
| UperNet | Swin-S | 512x512  | ImageNet-1K | 224x224 | 16          | 160000   | 6.17        | 14.72              | 47.72 | 49.24             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015.log.json)     |
| UperNet | Swin-B | 512x512 | ImageNet-1K | 224x224 | 16           | 160000   | 7.61        | 12.65              | 47.99 | 49.57             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340.log.json)     |
| UperNet | Swin-B | 512x512  | ImageNet-22K | 224x224 | 16          | 160000   | -        | -              | 50.31 | 51.9             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650.log.json)     |
| UperNet | Swin-B | 512x512  | ImageNet-1K | 384x384 | 16          | 160000   | 8.52        | 12.10              | 48.35 | 49.65             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K_20210531_132020-05b22ea4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K_20210531_132020.log.json)     |
| UperNet | Swin-B | 512x512  | ImageNet-22K | 384x384 | 16          | 160000   | -        | -              | 50.76 | 52.4             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459.log.json)     |
