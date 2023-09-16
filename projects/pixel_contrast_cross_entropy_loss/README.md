# Pixel contrast cross entropy loss

[Exploring Cross-Image Pixel Contrast for Semantic Segmentation](https://arxiv.org/pdf/2101.11939.pdf)

## Description

This is an implementation of **pixel contrast cross entropy loss**

[Official Repo](https://github.com/tfzhou/ContrastiveSeg)

## Abstract

Current semantic segmentation methods focus only on mining “local” context, i.e., dependencies between pixels within individual images, by context-aggregation modules (e.g., dilated convolution, neural attention) or structureaware optimization criteria (e.g., IoU-like loss). However, they ignore “global” context of the training data, i.e., rich semantic relations between pixels across different images. Inspired by the recent advance in unsupervised contrastive representation learning, we propose a pixel-wise contrastive framework for semantic segmentation in the fully supervised setting. The core idea is to enforce pixel embeddings belonging to a same semantic class to be more similar than embeddings from different classes. It raises a pixel-wise metric learning paradigm for semantic segmentation, by explicitly exploring the structures of labeled pixels, which are long ignored in the field. Our method can be effortlessly incorporated into existing segmentation frameworks without extra overhead during testing.

We experimentally show that, with famous segmentation models (i.e., DeepLabV3, HRNet, OCR) and backbones (i.e., ResNet, HRNet), our method brings consistent performance improvements across diverse datasets (i.e., Cityscapes, PASCALContext, COCO-Stuff).

## Usage

Here the configs for HRNet-W18 and HRNet-W48 with pixel_contrast_cross_entropy_loss on cityscapes dataset are provided.

After putting Cityscapes dataset into "mmsegmentation/data/" dir, train the network by:

```python
python tools/train.py projects/pixel_contrast_cross_entropy_loss/configs/fcn_hrcontrast48_4xb2-40k_cityscapes-512x1024.py
```

## Citation

```bibtex
@inproceedings{Wang_2021_ICCV,
    author    = {Wang, Wenguan and Zhou, Tianfei and Yu, Fisher and Dai, Jifeng and Konukoglu, Ender and Van Gool, Luc},
    title     = {Exploring Cross-Image Pixel Contrast for Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
    pages     = {7303-7313}
}
```
