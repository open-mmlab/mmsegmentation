# VPD

> [Unleashing Text-to-Image Diffusion Models for Visual Perception](https://arxiv.org/abs/2303.02153)

## Introduction

<!-- [ALGORITHM] -->

<a href = "https://github.com/wl-zhao/VPD">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

Diffusion models (DMs) have become the new trend of generative models and have demonstrated a powerful ability of conditional synthesis. Among those, text-to-image diffusion models pre-trained on large-scale image-text pairs are highly controllable by customizable prompts. Unlike the unconditional generative models that focus on low-level attributes and details, text-to-image diffusion models contain more high-level knowledge thanks to the vision-language pre-training. In this paper, we propose VPD (Visual Perception with a pre-trained Diffusion model), a new framework that exploits the semantic information of a pre-trained text-to-image diffusion model in visual perception tasks. Instead of using the pre-trained denoising autoencoder in a diffusion-based pipeline, we simply use it as a backbone and aim to study how to take full advantage of the learned knowledge. Specifically, we prompt the denoising decoder with proper textual inputs and refine the text features with an adapter, leading to a better alignment to the pre-trained stage and making the visual contents interact with the text prompts. We also propose to utilize the cross-attention maps between the visual features and the text features to provide explicit guidance. Compared with other pre-training methods, we show that vision-language pre-trained diffusion models can be faster adapted to downstream visual perception tasks using the proposed VPD. Extensive experiments on semantic segmentation, referring image segmentation and depth estimation demonstrates the effectiveness of our method. Notably, VPD attains 0.254 RMSE on NYUv2 depth estimation and 73.3% oIoU on RefCOCO-val referring image segmentation, establishing new records on these two benchmarks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/open-mmlab/mmsegmentation/assets/26127467/88f5752d-7fe2-4cb0-a284-8ee0680e29cd" width="80%"/>
</div>

## Usage

To run training or inference with VPD model, please install the required packages via

```sh
pip install -r requirements/albu.txt
pip install -r requirements/optional.txt
```

## Results and models

### NYU

| Method | Backbone              | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device | RMSE  | d1    | d2    | d3    | REL   | log_10 | config                                                                                                      | download                                                                                                                                                                                                                     |
| ------ | --------------------- | --------- | ------- | -------- | -------------- | ------ | ----- | ----- | ----- | ----- | ----- | ------ | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VPD    | Stable-Diffusion-v1-5 | 480x480   | 25000   | -        | -              | A100   | 0.253 | 0.964 | 0.995 | 0.999 | 0.069 | 0.030  | [config](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/vpd/vpd_sd_4xb8-25k_nyu-480x480.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vpd/vpd_sd_4xb8-25k_nyu-480x480_20230908-66144bc4.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vpd/vpd_sd_4xb8-25k_nyu-480x480_20230908.json) |
| VPD    | Stable-Diffusion-v1-5 | 512x512   | 25000   | -        | -              | A100   | 0.258 | 0.963 | 0.995 | 0.999 | 0.072 | 0.031  | [config](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/vpd/vpd_sd_4xb8-25k_nyu-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/vpd/vpd_sd_4xb8-25k_nyu-512x512_20230918-60cefcff.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/vpd/vpd_sd_4xb8-25k_nyu-512x512_20230918.json) |

## Citation

```bibtex
@article{zhao2023unleashing,
  title={Unleashing Text-to-Image Diffusion Models for Visual Perception},
  author={Zhao, Wenliang and Rao, Yongming and Liu, Zuyan and Liu, Benlin and Zhou, Jie and Lu, Jiwen},
  journal={ICCV},
  year={2023}
}
```
