# SAN

> [Side Adapter Network for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2302.12242)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/MendelXu/SAN">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

This paper presents a new framework for open-vocabulary semantic segmentation with the pre-trained vision-language model, named Side Adapter Network (SAN). Our approach models the semantic segmentation task as a region recognition problem. A side network is attached to a frozen CLIP model with two branches: one for predicting mask proposals, and the other for predicting attention bias which is applied in the CLIP model to recognize the class of masks. This decoupled design has the benefit CLIP in recognizing the class of mask proposals. Since the attached side network can reuse CLIP features, it can be very light. In addition, the entire network can be trained end-to-end, allowing the side network to be adapted to the frozen CLIP model, which makes the predicted mask proposals CLIP-aware. Our approach is fast, accurate, and only adds a few additional trainable parameters. We evaluate our approach on multiple semantic segmentation benchmarks. Our method significantly outperforms other counterparts, with up to 18 times fewer trainable parameters and 19 times faster inference speed. We hope our approach will serve as a solid baseline and help ease future research in open-vocabulary semantic segmentation.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/MendelXu/SAN/blob/main/resources/arch.png" width="800"/>
</div>

## Results and models

### COCO-Stuff164k

| Method | Backbone | Pretrained   | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device | mIoU  | mIoU(ms+flip) | config                                                                                                   | download                                                                                                                                                                                    |
| ------ | -------- | ------------ | --------- | ------- | -------- | -------------- | ------ | ----- | ------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SAN    | ViT-B_16 | CLIP_ViT-B16 | 640x640   | 60000   | 12.61    | -              | V100   | 41.93 | 41.77         | https://github.com/open-mmlab/mmsegmentation/blob/main/configs/san/san-vit-b16_coco-stuff164k-640x640.py | [model](https://download.openmmlab.com/mmsegmentation/v0.5/san/san-vit-b16_20230906-fd0a7684.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/san/san-vit-b16_20230906.log) |
| SAN    | ViT-L_14 | CLIP_ViT-L14 | 640x640   | 60000   | 22.84    | -              | V100   | 45.78 | 43.99         | https://github.com/open-mmlab/mmsegmentation/blob/main/configs/san/san-vit-l14_coco-stuff164k-640x640.py | [model](https://download.openmmlab.com/mmsegmentation/v0.5/san/san-vit-l14_20230907-a11e098f.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/san/san-vit-l14_20230907.log) |

## Notes

git push
The pretrained weights in config files are converted from open_clip models using tools/model_converters/clip2mmseg.py.

## Citation

```bibtex
@inproceedings{xu2023side,
  title={Side adapter network for open-vocabulary semantic segmentation},
  author={Xu, Mengde and Zhang, Zheng and Wei, Fangyun and Hu, Han and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2945--2954},
  year={2023}
}
```
