# ImageNet-S Dataset for Large-scale Unsupervised/Semi-supervised Semantic Segmentation

Large-scale Unsupervised Semantic Segmentation (TPAMI 2022)

<a href="https://lusseg.github.io/">Project page</a> 

<a href="https://arxiv.org/abs/2106.03149">Paper link</a> 

<a href="https://paperswithcode.com/dataset/imagenet-s">PaperWithCode</a>

## Introduction

![image](https://user-images.githubusercontent.com/20515144/149651945-94501ffc-78c0-41be-a1d9-b3bfb3253370.png)
<!-- [ALGORITHM] -->

<a href="https://github.com/LUSSeg/ImageNetSegModel">Official Repo</a>

<a href="blob/main/mmseg/datasets/imagenets.py#L92">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Powered by the ImageNet dataset, unsupervised learning on large-scale data has made significant advances for classification tasks. There are two major challenges to allowing such an attractive learning modality for segmentation tasks: i) a large-scale benchmark for assessing algorithms is missing; ii) unsupervised shape representation learning is difficult. We propose a new problem of large-scale unsupervised semantic segmentation (LUSS) with a newly created benchmark dataset to track the research progress. Based on the ImageNet dataset, we propose the ImageNet-S dataset with 1.2 million training images and 50k high-quality semantic segmentation annotations for evaluation. Our benchmark has a high data diversity and a clear task objective. We also present a simple yet effective baseline method that works surprisingly well for LUSS. In addition, we benchmark related un/weakly/fully supervised methods accordingly, identifying the challenges and possible directions of LUSS.

## Apps and Sourcecode
- Unsupervised semantic segmentation: [PASS](https://github.com/LUSSeg/PASS)
- Semi-supervised semantic segmentation: [ImageNetSegModel](https://github.com/LUSSeg/ImageNetSegModel)

**Here we also provide the sourecode for semi-supervised semantic segmentation.**
## Image Numbers
The ImageNet-S dataset contains 1183322 training, 12419 validation, and 27423 testing images from 919 categories. We annotate 39842 val/test images and 9190 training images with precise pixel-level masks.

| Dataset | category | train   | train-semi | val   | test  |
|------------------|----------|---------|---------|-------|-------|
| ImageNet-S50  | 50       | 64431   | 500 | 752   | 1682  |
| ImageNet-S300 | 300      | 384862  | 3000 | 4097  | 9088  |
| ImageNet-S        | 919      | 1183322 | 9190 | 12419 | 27423 |

## Online benchmark
More details about online benchmark is on the [project page](https://LUSSeg.github.io/).
* Fully unsupervised protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1317)
* Distance matching protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1315)
* Semi-supervised protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1318)
* Free protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1316)

**How to submit results to online benchmark?**

First set the data config in `configs/_base_/datasets/imagenets.py`:

```python
test=dict(
        type=dataset_type,
        subset=subset,
        data_root=data_root,
        img_dir='test',
        ann_dir='test-segmentation',
        pipeline=test_pipeline)
```

Then please generate the png files. 
Finally, please generate the required `method.txt` and zip the files.
The generated zip file cold be submit to the evaluation server. 
Note that the `method`, `arch`, `train_data`, `train_scheme`, `link` and `description` 
are the description of your method and are set as none by default.

```shell
python ./tools/test.py [CONFIG] \
    [CHECKPOINT] \
    --format-only --eval-options "imgfile_prefix=./imagenets"

cd configs/imagenets

# generate the required `method.txt` and zip the files.

python imagenets_submit.py --imgfile_prefix ./imagenets \
--method [Method name in method description file(.txt).] \
--arch [The model architecture in method description file(.txt).] \
--train_data [Training data in method description file(.txt).] \
--train_scheme [Training scheme in method description file(.txt), e.g., SSL, Sup, SSL+Sup.] \
--link [Paper/project link in method description file(.txt).] \
--description [Method description in method description file(.txt).]
```

## Citation

```bibtex
@article{gao2022luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal=TPAMI,
  year={2022}
}
```

## Semi-supervised semantic segmentation

To finetune with different pre-trained models, please convert keys following [`vit`](../vit/README.md).

### Results and models

| Method | Backbone | pre-training epochs | pre-training mode | Crop Size | Lr schd (Epoch) | Mem (GB) | Inf time (fps) | mIoU | pre-trained                                                                                                         | config                                                                                                                                           | download                 |
| ------ | -------- | ------------------- | ----------------- | --------- | ------: | -------- | -------------- | ---: | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| MAE    | ViT-B/16 | 1600                | SSL               | 224x224   |    100 |          |                |   | [pre-trained](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)                                | [config](https://github.com/LUSSeg/mmsegmentation/blob/master/configs/imagenets/fcn_mae-base_pretrained_fp16_8x32_224x224_100ep_imagenets919.py)   | [model](<>) \| [log](<>) |
| MAE    | ViT-B/16 | 1600                | SSL+Sup           | 224x224   |    100 |          |                |   | [pre-trained](https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth)                               | [config](https://github.com/LUSSeg/mmsegmentation/blob/master/configs/imagenets/fcn_mae-base_finetuned_fp16_8x32_224x224_100ep_imagenets919.py)    | [model](<>) \| [log](<>) |
| SERE   | ViT-S/16 | 100                 | SSL               | 224x224   |    100 |          |                |  41.0 | [pre-trained](https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_pretrained_vit_small_ep100.pth) | [config](https://github.com/LUSSeg/mmsegmentation/blob/master/configs/imagenets/fcn_sere-small_pretrained_fp16_8x32_224x224_100ep_imagenets919.py) | [model](<>) \| [log](<>) |
| SERE   | ViT-S/16 | 100                 | SSL+Sup           | 224x224   |    100 |          |                | 59.4  | [pre-trained](https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_finetuned_vit_small_ep100.pth)  | [config](https://github.com/LUSSeg/mmsegmentation/blob/master/configs/imagenets/fcn_sere-small_finetuned_fp16_8x32_224x224_100ep_imagenets919.py)  | [model](<>) \| [log](<>) |
