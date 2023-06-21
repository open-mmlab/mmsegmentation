# ConvNeXt

> [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/facebookresearch/ConvNeXt">Official Repo</a>

<a href="https://github.com/open-mmlab/mmclassification/blob/v0.20.1/mmcls/models/backbones/convnext.py#L133">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/148624004-e9581042-ea4d-4e10-b3bd-42c92b02053b.png" width="90%"/>
</div>

### Usage

- ConvNeXt backbone needs to install [MMClassification](https://github.com/open-mmlab/mmclassification) first, which has abundant backbones for downstream tasks.

```shell
pip install mmcls>=0.20.1
```

### Pre-trained Models

The pre-trained models on ImageNet-1k or ImageNet-21k are used to fine-tune on the downstream tasks.

|     Model     | Training Data | Params(M) | Flops(G) |                                                                     Download                                                                     |
| :-----------: | :-----------: | :-------: | :------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| ConvNeXt-T\*  |  ImageNet-1k  |   28.59   |   4.46   | [model](https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth)  |
| ConvNeXt-S\*  |  ImageNet-1k  |   50.22   |   8.69   | [model](https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth) |
| ConvNeXt-B\*  |  ImageNet-1k  |   88.59   |  15.36   | [model](https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth)  |
| ConvNeXt-B\*  | ImageNet-21k  |   88.59   |  15.36   |        [model](https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth)        |
| ConvNeXt-L\*  | ImageNet-21k  |  197.77   |  34.37   |       [model](https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth)        |
| ConvNeXt-XL\* | ImageNet-21k  |  350.20   |  60.93   |       [model](https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth)       |

*Models with* are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation#results-and-fine-tuned-models).\*

## Results and models

### ADE20K

| Method  | Backbone    | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device | mIoU  | mIoU(ms+flip) | config                                                                                                                                    | download                                                                                                                                                                                                                                                                                                                                                                                             |
| ------- | ----------- | --------- | ------- | -------- | -------------- | ------ | ----- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UPerNet | ConvNeXt-T  | 512x512   | 160000  | 4.23     | 19.90          | V100   | 46.11 | 46.62         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/convnext/convnext-tiny_upernet_8xb2-amp-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553.log.json)         |
| UPerNet | ConvNeXt-S  | 512x512   | 160000  | 5.16     | 15.18          | V100   | 48.56 | 49.02         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/convnext/convnext-small_upernet_8xb2-amp-160k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_small_fp16_512x512_160k_ade20k/upernet_convnext_small_fp16_512x512_160k_ade20k_20220227_131208-1b1e394f.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_small_fp16_512x512_160k_ade20k/upernet_convnext_small_fp16_512x512_160k_ade20k_20220227_131208.log.json)     |
| UPerNet | ConvNeXt-B  | 512x512   | 160000  | 6.33     | 14.41          | V100   | 48.71 | 49.54         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/convnext/convnext-base_upernet_8xb2-amp-160k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227.log.json)         |
| UPerNet | ConvNeXt-B  | 640x640   | 160000  | 8.53     | 10.88          | V100   | 52.13 | 52.66         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/convnext/convnext-base_upernet_8xb2-amp-160k_ade20k-640x640.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_640x640_160k_ade20k/upernet_convnext_base_fp16_640x640_160k_ade20k_20220227_182859-9280e39b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_640x640_160k_ade20k/upernet_convnext_base_fp16_640x640_160k_ade20k_20220227_182859.log.json)         |
| UPerNet | ConvNeXt-L  | 640x640   | 160000  | 12.08    | 7.69           | V100   | 53.16 | 53.38         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/convnext/convnext-large_upernet_8xb2-amp-160k_ade20k-640x640.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_large_fp16_640x640_160k_ade20k/upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532-e57aa54d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_large_fp16_640x640_160k_ade20k/upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532.log.json)     |
| UPerNet | ConvNeXt-XL | 640x640   | 160000  | 26.16\*  | 6.33           | V100   | 53.58 | 54.11         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/convnext/convnext-xlarge_upernet_8xb2-amp-160k_ade20k-640x640.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_xlarge_fp16_640x640_160k_ade20k/upernet_convnext_xlarge_fp16_640x640_160k_ade20k_20220226_080344-95fc38c2.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_xlarge_fp16_640x640_160k_ade20k/upernet_convnext_xlarge_fp16_640x640_160k_ade20k_20220226_080344.log.json) |

Note:

- `Mem (GB)` with * is collected when `cudnn_benchmark=True`, and hardware is V100.

## Citation

```bibtex
@article{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
