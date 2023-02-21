# HRFormer

[HRFormer: High-Resolution Transformer for Dense Prediction](https://arxiv.org/pdf/2110.09408.pdf)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/HRNet/HRFormer">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.21.0/mmseg/models/backbones/hrformer.py#L562">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

We present a High-Resolution Transformer (HRFormer) that learns high-resolution representations for dense prediction tasks, in contrast to the original Vision Transformer that produces low-resolution representations and has high memory and computational cost. We take advantage of the multi-resolution parallel design introduced in high-resolution convolutional networks (HRNet), along with local-window self-attention that performs self-attention over small non-overlapping image windows, for improving the memory and computation efficiency. In addition, we introduce a convolution into the FFN to exchange information across the disconnected image windows. We demonstrate the effectiveness of the High- Resolution Transformer on both human pose estimation and semantic segmentation tasks, e.g., HRFormer outperforms Swin transformer by 1.3 AP on COCO pose estimation with 50% fewer parameters and 30% fewer FLOPs.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/41846794/160131437-46a06d14-549b-41c4-9fc0-e4785fb63468.png" width="70%"/>
</div>


```bibtex
@article{YuanFHLZCW21,
  title={HRFormer: High-Resolution Transformer for Dense Prediction},
  author={Yuan, Yuhui and Fu, Rao and Huang, Lang and Lin, Weihong and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  booktitle={NeurIPS},
  year={2021}
}
```

## Usage

To use the pre-trained models from the [official repository](https://github.com/HRNet/HRFormer), it is necessary to convert keys.

We provide a script [`hrformer2mmseg.py`](../../tools/model_converters/hrformer2mmseg.py) in the tools directory to convert the keys of the pre-trained models from [the official repo](https://github.com/HRNet/HRFormer/tree/main/cls) to MMSegmentation style.

```shell
python tools/model_converters/hrformer2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

E.g.

```shell
python tools/model_converters/hrformer2mmseg.py pretrain/htr_base_origin.pth pretrain/htr_base.pth
```

## Results and models

### Cityscapes

| Method | Backbone           | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) |                                                       config                                                                         | download                                                                                                                                                                                                                                                                                                                                   |
| ------ | ------------------ | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| OCRNet |     HRFormer-S     | 512x1024  |  80000  |   11.54  |      2.24      | 80.68 |    81.65      | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrformer/ocrnet_hrformer-s_4x2_512x1024_80k_cityscapes.py) |  [model]() &#124; [log]()     |
| OCRNet |     HRFormer-B     | 512x1024  |  80000  |   23.80  |      1.15      | 82.07 |    82.69      | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrformer/ocrnet_hrformer-b_4x2_512x1024_80k_cityscapes.py) |  [model]() &#124; [log]()     |
