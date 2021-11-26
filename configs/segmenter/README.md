# Segmenter: Transformer for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/rstrudel/segmenter">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.21.0/mmseg/models/backbones/mit.py#L246">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/2105.05633">Segmenter: Transformer for Semantic Segmentation (ICCV'2021)</a></summary>

```latex
@article{strudel2021Segmenter,
  title={Segmenter: Transformer for Semantic Segmentation},
  author={Strudel, Robin and Ricardo, Garcia, and Laptev, Ivan and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2105.05633},
  year={2021}
}
```

</details>

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`vitjax2mmseg.py`](../../tools/model_converters/vitjax2mmseg.py) in the tools directory to convert the key of models from [ViT-AugReg](https://github.com/google-research/vision_transformer) to MMSegmentation style.

```shell
python tools/model_converters/vitjax2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

E.g.

```shell
python tools/model_converters/vitjax2mmseg.py \
https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz \
pretrain/vit_small_patch16_384.pth
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ---------- | ------- | -------- | --- | --- | -------------- | ----- |
| Segmenter-Linear | ViT-S_16 | 512x512 | 160000   | 999        | 34.78              | 45.57 | 45.69            | [config]() | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530.log.json) |
| Segmenter-Mask | ViT-S_16 | 512x512 | 160000   | 999        | 29.80              | 45.96 | 46.51            | [config]() | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530.log.json) |
| Segmenter-Linear | ViT-B_16 | 512x512 | 160000   | 999        | 17.34              | 48.69 | 48.71            | [config]() | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530.log.json) |
| Segmenter-Mask | ViT-B_16 |512x512 |  160000   | 999        | 14.67              | 48.69 | 49.51            | [config]()  | [model]() &#124; [log]()     |
