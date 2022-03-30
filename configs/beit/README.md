# BEiT

[BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/microsoft/unilm/tree/master/beit">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.23.0/mmseg/models/backbones/beit.py#1404">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. Experimental results on image classification and semantic segmentation show that our model achieves competitive results with previous pre-training methods. For example, base-size BEiT achieves 83.2% top-1 accuracy on ImageNet-1K, significantly outperforming from-scratch DeiT training (81.8%) with the same setup. Moreover, large-size BEiT obtains 86.3% only using ImageNet-1K, even outperforming ViT-L with supervised pre-training on ImageNet-22K (85.2%). The code and pretrained models are available at [this https URL](https://github.com/microsoft/unilm/tree/master/beit).

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/93248678/160155758-781c9a45-b1d7-4530-9015-88eca6645006.png" width="70%"/>
</div>

## Citation

```bibtex
@inproceedings{beit,
      title={{BEiT}: {BERT} Pre-Training of Image Transformers},
      author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=p-BhZSz59o4}
}
```

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`beit2mmseg.py`](../../tools/model_converters/beit2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/microsoft/unilm/tree/master/beit/semantic_segmentation) to MMSegmentation style.

```shell
python tools/model_converters/beit2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

E.g.

```shell
python tools/model_converters/beit2mmseg.py https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth pretrain/beit_base_patch16_224_pt22k_ft22k.pth
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

In our default setting, pretrained models could be defined below:

  | pretrained models | original models |
  | ------ | -------- |
  |BEiT_base.pth | ['BEiT_base'](https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth) |
  |BEiT_large.pth | ['BEiT_large'](https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22k.pth) |

Verify the single-scale results of the model:

```shell
sh tools/dist_test.sh \
configs/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py \
upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth $GPUS --eval mIoU
```

Since relative position embedding requires the input length and width to be equal, the sliding window is adopted for multi-scale inference. So we set min_size=640, that is, the shortest edge is 640. So the multi-scale inference of config is performed separately, instead of '--aug-test'. For multi-scale inference:

```shell
sh tools/dist_test.sh \
configs/beit/upernet_beit-large_fp16_640x640_160k_ade20k_ms.py \
upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth $GPUS --eval mIoU
```

## Results and models

### ADE20K

| Method | Backbone | Crop Size | pretrain | pretrain img size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ---------- | ------- | -------- | --- | --- | -------------- | ----- | ------------: | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UperNet | BEiT-B | 640x640 | ImageNet-22K | 224x224 | 16          | 160000   | 15.88        | 2.00              | 53.08 | 53.84            | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/beit/upernet_beit-base_8x2_640x640_160k_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-base_8x2_640x640_160k_ade20k/upernet_beit-base_8x2_640x640_160k_ade20k-eead221d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-base_8x2_640x640_160k_ade20k/upernet_beit-base_8x2_640x640_160k_ade20k.log.json)     |
| UperNet | BEiT-L | 640x640 | ImageNet-22K | 224x224 | 8           | 320000   | 22.64        | 0.96              | 56.33 | 56.84             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.log.json)     |
