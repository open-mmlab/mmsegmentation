# CAT-Seg

> [CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2303.11797)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/KU-CVLAB/CAT-Seg">Official Repo</a>

<a href="https://github.com/SheffieldCao/mmsegmentation/blob/support-cat-seg/mmseg/models/necks/cat_aggregator.py">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Existing works on open-vocabulary semantic segmentation have utilized large-scale vision-language models, such as CLIP, to leverage their exceptional open-vocabulary recognition capabilities. However, the problem of transferring these capabilities learned from image-level supervision to the pixel-level task of segmentation and addressing arbitrary unseen categories at inference makes this task challenging. To address these issues, we aim to attentively relate objects within an image to given categories by leveraging relational information among class categories and visual semantics through aggregation, while also adapting the CLIP representations to the pixel-level task. However, we observe that direct optimization of the CLIP embeddings can harm its open-vocabulary capabilities. In this regard, we propose an alternative approach to optimize the imagetext similarity map, i.e. the cost map, using a novel cost aggregation-based method. Our framework, namely CATSeg, achieves state-of-the-art performance across all benchmarks. We provide extensive ablation studies to validate our choices. [Project page](https://ku-cvlab.github.io/CAT-Seg).

<!-- [IMAGE] -->

<div align=center >
<img alt="CAT-Seg" src="https://github.com/open-mmlab/mmsegmentation/assets/49406546/d54674bb-52ae-4a20-a168-e25d041111e8"/>
CAT-Seg model structure
</div>

## Usage

CAT-Seg model training needs pretrained `CLIP` model. We have implemented `ViT-B` and `ViT-L` based `CLIP` model. To further use `ViT-bigG` or `ViT-H` ones, you need additional dependencies. Please install [open_clip](https://github.com/mlfoundations/open_clip) first. The pretrained `CLIP` model state dicts are loaded from [Huggingface-OpenCLIP](https://huggingface.co/models?library=open_clip). **If you come up with `ConnectionError` when downloading CLIP weights**, you can manually download them from the given repo and use `custom_clip_weights=/path/to/you/folder` of backbone in config file. Related tools are as shown in [requirements/optional.txt](requirements/optional.txt):

```shell
pip install ftfy==6.0.1
pip install huggingface-hub
pip install regex
```

In addition to the necessary [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md), you also need class texts for clip text encoder. Please download the class text json file first [cls_texts](https://github.com/open-mmlab/mmsegmentation/files/11714914/cls_texts.zip) and arrange the folder as follows:

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   ├── VOC2010
│   │   ├── VOCaug
│   ├── ade
│   ├── coco_stuff164k
│   ├── coco.json
│   ├── pc59.json
│   ├── pc459.json
│   ├── ade150.json
│   ├── ade847.json
│   ├── voc20b.json
│   ├── voc20.json
```

```shell
# setup PYTHONPATH
export PYTHONPATH=`pwd`:$PYTHONPATH
# run evaluation
mim test mmsegmentation ${CONFIG} --checkpoint ${CHECKPOINT} --launcher pytorch --gpus=8
```

## Results and models

### ADE20K-150-ZeroShot

| Method  | Backbone      | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device  | mIoU | mIoU(ms+flip) |                                                                                      config | download                                                                                                                                      |
| ------- | ------------- | --------- | ------- | -------: | -------------- | ------- | ---- | ------------: | ------------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------- |
| CAT-Seg | R-101 & ViT-B | 384x384   | 80000   |        - | -              | RTX3090 | 27.2 |             - | [config](./configs/cat_seg/catseg_vitb-r101_4xb1-warmcoslr2e-4-adamw-80k_ade20k-384x384.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/cat_seg/catseg_vitb-r101_4xb1-warmcoslr2e-4-adamw-80k_ade20k-384x384-54194d72.pth) |

Note:

- All experiments of CAT-Seg are implemented with 4 RTX3090 GPUs, except the last one with pretrained ViT-bigG CLIP model (GPU Memory insufficient, you may need A100).
- Due to the feature size bottleneck of the CLIP image encoder, the inference and testing can only be done under `slide` mode, the inference time is longer since the test size is much more bigger that training size of `(384, 384)`.
- The ResNet backbones utilized in CAT-Seg models are standard `ResNet` rather than `ResNetV1c`.
- The zero-shot segmentation results on PASCAL VOC and ADE20K are from the original paper. Our results are coming soon. We appreatiate your contribution!
- In additional to zero-shot segmentation performance results, we also provided the evaluation results on the `val2017` set of **COCO-stuff164k** for reference, which is the training dataset of CAT-Seg. The testing was done **without TTA**.
- The number behind the dataset name is the category number for segmentation evaluation (except training data **COCO-stuff 164k**). **PASCAL VOC-20b** defines the "background" as classes present in **PASCAL-Context-59** but not in **PASCAL VOC-20**.

## Citation

```bibtex
@inproceedings{cheng2021mask2former,
  title={CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation},
  author={Seokju Cho and Heeseong Shin and Sunghwan Hong and Seungjun An and Seungjun Lee and Anurag Arnab and Paul Hongsuck Seo and Seungryong Kim},
  journal={CVPR},
  year={2023}
}
```
