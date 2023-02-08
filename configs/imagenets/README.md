# ImageNet-S Dataset for Large-scale Semantic Segmentation

Large-scale Unsupervised Semantic Segmentation (TPAMI 2022) （<a href="https://lusseg.github.io/">Project page</a> | <a href="https://arxiv.org/abs/2106.03149">Paper link</a> | <a href="https://github.com/LUSSeg/ImageNet-S">Dataset Download</a> )

![image](https://user-images.githubusercontent.com/20515144/149651945-94501ffc-78c0-41be-a1d9-b3bfb3253370.png)

## About ImageNet-S

<!-- [ABSTRACT] -->

Powered by the ImageNet dataset, unsupervised learning on large-scale data has made significant advances for classification tasks. There are two major challenges to allowing such an attractive learning modality for segmentation tasks: i) a large-scale benchmark for assessing algorithms is missing; ii) unsupervised shape representation learning is difficult. We propose a new problem of large-scale unsupervised semantic segmentation (LUSS) with a newly created benchmark dataset to track the research progress. Based on the ImageNet dataset, we propose the ImageNet-S dataset has 1.2 million training images and 50k high-quality semantic segmentation annotations to support unsupervised/semi-supervised semantic segmentation on the ImageNet dataset.

The ImageNet-S dataset contains 1183322 training, 12419 validation, and 27423 testing images from 919 categories. We annotate 39842 val/test images and 9190 training images with precise pixel-level masks.

| Dataset       | category | train   | train-semi | val   | test  |
| ------------- | -------- | ------- | ---------- | ----- | ----- |
| ImageNet-S50  | 50       | 64431   | 500        | 752   | 1682  |
| ImageNet-S300 | 300      | 384862  | 3000       | 4097  | 9088  |
| ImageNet-S    | 919      | 1183322 | 9190       | 12419 | 27423 |

<!-- [ALGORITHM] -->

<a href="blob/main/mmseg/datasets/imagenets.py#L92">Code Snippet</a>

## Semi-supervised Semantic Segmentation Results

<a href="https://paperswithcode.com/dataset/imagenet-s">**PaperWithCode Leaderboard**</a>

| Method | Backbone | Pre-training epochs | Pre-training mode | Crop Size | Lr schd (Epoch) | mIoU | mIoU (test) | Pre-trained                                                                                                         | Config                                                                                                                                                 | Download                 |
| ------ | -------- | ------------------- | ----------------- | --------- | --------------: | ---: | ----------: | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| MAE    | ViT-B/16 | 1600                | SSL               | 224x224   |             100 | 40.0 |        40.3 | [pre-trained](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)                                | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/imagenets/fcn_mae-base_pretrained_fp16_8x32_224x224_100ep_imagenets919.py)   | [model](<>) \| [log](<>) |
| MAE    | ViT-B/16 | 1600                | SSL+Sup           | 224x224   |             100 | 61.6 |        61.2 | [pre-trained](https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth)                               | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/imagenets/fcn_mae-base_finetuned_fp16_8x32_224x224_100ep_imagenets919.py)    | [model](<>) \| [log](<>) |
| SERE   | ViT-S/16 | 100                 | SSL               | 224x224   |             100 | 41.0 |        40.5 | [pre-trained](https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_pretrained_vit_small_ep100.pth) | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/imagenets/fcn_sere-small_pretrained_fp16_8x32_224x224_100ep_imagenets919.py) | [model](<>) \| [log](<>) |
| SERE   | ViT-S/16 | 100                 | SSL+Sup           | 224x224   |             100 | 59.4 |        59.0 | [pre-trained](https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_finetuned_vit_small_ep100.pth)  | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/imagenets/fcn_sere-small_finetuned_fp16_8x32_224x224_100ep_imagenets919.py)  | [model](<>) \| [log](<>) |

### Training

We provide the training configs using ViT models. The pretraining weights of ViT backbone should be converted following [vit](../vit/README.md):

```shell
python tools/model_converters/vit2mmseg.py https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth pretrain/mae_pretrain_vit_base_mmcls.pth
```

### Evaluation

- The mmsegmentation supports the evaluation on the val set.
- To evaluate the test set, please submit the prediction to the online benchmarks: ([Fully unsupervised](https://codalab.lisn.upsaclay.fr/competitions/1317)|[Distance matching](https://codalab.lisn.upsaclay.fr/competitions/1315)|[Semi-supervised](https://codalab.lisn.upsaclay.fr/competitions/1318)|[Free](https://codalab.lisn.upsaclay.fr/competitions/1316)).
  More details about online benchmark is on the [project page](https://LUSSeg.github.io/).

#### Submit test set results to online benchmarks:

1. Change the evaluation dataset from 'val set' to 'test set' in the data config file `configs/_base_/datasets/imagenets.py`:

```python
test=dict(
  type=dataset_type,
  subset=subset,
  data_root=data_root,
  img_dir='test',
  ann_dir='test-segmentation',
  pipeline=test_pipeline)
```

2. Generate the prediction results of the test set.

```shell
python ./tools/test.py [CONFIG] \
    [CHECKPOINT] \
    --format-only --eval-options "imgfile_prefix=[path/to/the/saved/test/prediction/results.]"
```

3. Generate the method description file `method.txt` and zip the prediction results.
   The generated zip file can be submit to the online evaluation server.

```shell
cd configs/imagenets

python imagenets_submit.py --imgfile_prefix [path/to/the/saved/test/prediction/results.] \
--method [Method name.] \
--arch [The model architecture.] \
--train_data [Training data.] \
--train_scheme [Training scheme description, e.g., SSL, Sup, SSL+Sup.] \
--link [Paper/project link.] \
--description [Method description.]
```

Note that the `method`, `arch`, `train_data`, `train_scheme`, `link` and `description`
are the description of your method and are set as none by default.

## Other Apps and Sourcecode using ImageNet-S

- Unsupervised semantic segmentation: [PASS](https://github.com/LUSSeg/PASS)
- Semi-supervised semantic segmentation: [ImageNetSegModel](https://github.com/LUSSeg/ImageNetSegModel)

## Citation

```bibtex
@article{gao2022luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal=TPAMI,
  year={2022}
}
```
