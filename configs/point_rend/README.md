# PointRend

> [PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/point_head.py#L36">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

We present a new method for efficient high-quality image segmentation of objects and scenes. By analogizing classical computer graphics methods for efficient rendering with over- and undersampling challenges faced in pixel labeling tasks, we develop a unique perspective of image segmentation as a rendering problem. From this vantage, we present the PointRend (Point-based Rendering) neural network module: a module that performs point-based segmentation predictions at adaptively selected locations based on an iterative subdivision algorithm. PointRend can be flexibly applied to both instance and semantic segmentation tasks by building on top of existing state-of-the-art models. While many concrete implementations of the general idea are possible, we show that a simple design already achieves excellent results. Qualitatively, PointRend outputs crisp object boundaries in regions that are over-smoothed by previous methods. Quantitatively, PointRend yields significant gains on COCO and Cityscapes, for both instance and semantic segmentation. PointRend's efficiency enables output resolutions that are otherwise impractical in terms of memory or computation compared to existing approaches. Code has been made available at [this https URL](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142902293-5db49cdd-4b1b-4940-9067-2acd6196c700.png" width="60%"/>
</div>

## Results and models

### Cityscapes

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                             | download                                                                                                                                                                                                                                                                                                                                                         |
| --------- | -------- | --------- | ------: | -------: | -------------- | ------ | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PointRend | R-50     | 512x1024  |   80000 |      3.1 | 8.48           | V100   | 76.47 | 78.13         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/point_rend/pointrend_r50_4xb2-80k_cityscapes-512x1024.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r50_512x1024_80k_cityscapes/pointrend_r50_512x1024_80k_cityscapes_20200711_015821-bb1ff523.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r50_512x1024_80k_cityscapes/pointrend_r50_512x1024_80k_cityscapes-20200715_214714.log.json)     |
| PointRend | R-101    | 512x1024  |   80000 |      4.2 | 7.00           | V100   | 78.30 | 79.97         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/point_rend/pointrend_r101_4xb2-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r101_512x1024_80k_cityscapes/pointrend_r101_512x1024_80k_cityscapes_20200711_170850-d0ca84be.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r101_512x1024_80k_cityscapes/pointrend_r101_512x1024_80k_cityscapes-20200715_214824.log.json) |

### ADE20K

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                         | download                                                                                                                                                                                                                                                                                                                                         |
| --------- | -------- | --------- | ------: | -------: | -------------- | ------ | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PointRend | R-50     | 512x512   |  160000 |      5.1 | 17.31          | V100   | 37.64 | 39.17         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/point_rend/pointrend_r50_4xb4-160k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r50_512x512_160k_ade20k/pointrend_r50_512x512_160k_ade20k_20200807_232644-ac3febf2.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r50_512x512_160k_ade20k/pointrend_r50_512x512_160k_ade20k-20200807_232644.log.json)     |
| PointRend | R-101    | 512x512   |  160000 |      6.1 | 15.50          | V100   | 40.02 | 41.60         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/point_rend/pointrend_r101_4xb4-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r101_512x512_160k_ade20k/pointrend_r101_512x512_160k_ade20k_20200808_030852-8834902a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r101_512x512_160k_ade20k/pointrend_r101_512x512_160k_ade20k-20200808_030852.log.json) |

## Citation

```bibtex
@inproceedings{kirillov2020pointrend,
  title={Pointrend: Image segmentation as rendering},
  author={Kirillov, Alexander and Wu, Yuxin and He, Kaiming and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={9799--9808},
  year={2020}
}
```
