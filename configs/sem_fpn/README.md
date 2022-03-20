# Semantic FPN

[Panoptic Feature Pyramid Networks](https://arxiv.org/abs/1901.02446)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/facebookresearch/detectron2">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/fpn_head.py#L12">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The recently introduced panoptic segmentation task has renewed our community's interest in unifying the tasks of instance segmentation (for thing classes) and semantic segmentation (for stuff classes). However, current state-of-the-art methods for this joint task use separate and dissimilar networks for instance and semantic segmentation, without performing any shared computation. In this work, we aim to unify these methods at the architectural level, designing a single network for both tasks. Our approach is to endow Mask R-CNN, a popular instance segmentation method, with a semantic segmentation branch using a shared Feature Pyramid Network (FPN) backbone. Surprisingly, this simple baseline not only remains effective for instance segmentation, but also yields a lightweight, top-performing method for semantic segmentation. In this work, we perform a detailed study of this minimally extended version of Mask R-CNN with FPN, which we refer to as Panoptic FPN, and show it is a robust and accurate baseline for both tasks. Given its effectiveness and conceptual simplicity, we hope our method can serve as a strong baseline and aid future research in panoptic segmentation.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142902694-03ed2131-9104-467b-ace1-c74c62fb7177.png" width="60%"/>
</div>

## Citation

```bibtex
@inproceedings{kirillov2019panoptic,
  title={Panoptic feature pyramid networks},
  author={Kirillov, Alexander and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6399--6408},
  year={2019}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FPN    | R-50     | 512x1024  |   80000 |      2.8 | 13.54          | 74.52 | 76.08         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/sem_fpn/fpn_r50_512x1024_80k_cityscapes.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/sem_fpn/fpn_r50_512x1024_80k_cityscapes/fpn_r50_512x1024_80k_cityscapes_20200717_021437-94018a0d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/sem_fpn/fpn_r50_512x1024_80k_cityscapes/fpn_r50_512x1024_80k_cityscapes-20200717_021437.log.json)     |
| FPN    | R-101    | 512x1024  |   80000 |      3.9 | 10.29          | 75.80 | 77.40         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/sem_fpn/fpn_r101_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/sem_fpn/fpn_r101_512x1024_80k_cityscapes/fpn_r101_512x1024_80k_cityscapes_20200717_012416-c5800d4c.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/sem_fpn/fpn_r101_512x1024_80k_cityscapes/fpn_r101_512x1024_80k_cityscapes-20200717_012416.log.json) |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                             | download                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FPN    | R-50     | 512x512   |  160000 |      4.9 | 55.77          | 37.49 | 39.09         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/sem_fpn/fpn_r50_512x512_160k_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/sem_fpn/fpn_r50_512x512_160k_ade20k/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/sem_fpn/fpn_r50_512x512_160k_ade20k/fpn_r50_512x512_160k_ade20k-20200718_131734.log.json)     |
| FPN    | R-101    | 512x512   |  160000 |      5.9 | 40.58          | 39.35 | 40.72         | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/sem_fpn/fpn_r101_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/sem_fpn/fpn_r101_512x512_160k_ade20k/fpn_r101_512x512_160k_ade20k_20200718_131734-306b5004.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/sem_fpn/fpn_r101_512x512_160k_ade20k/fpn_r101_512x512_160k_ade20k-20200718_131734.log.json) |
