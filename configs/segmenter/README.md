# Segmenter: Transformer for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/rstrudel/segmenter">Official Repo</a>

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
| Segmenter-Linear | ViT-S/16 | 512x512 | 160000   | mem        | fps              | ? | ?            | [config]()  | [model]() &#124; [log]()     |
| Segmenter-Mask | ViT-S/16 | 512x512 | 160000   | mem        | fps              | 45.96 | 46.51            | [config]()  | [model]() &#124; [log]()     |
| Segmenter-Linear | ViT-B/16 | 512x512 | 160000   | mem        | fps              | 48.69 | 48.71            | [config]()  | [model]() &#124; [log]()     |
| Segmenter-Mask | ViT-B/16 |512x512 |  160000   | mem        | fps              | 48.69 | 49.51            | [config]()  | [model]() &#124; [log]()     |
