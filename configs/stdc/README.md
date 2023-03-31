# STDC

> [Rethinking BiSeNet For Real-time Semantic Segmentation](https://arxiv.org/abs/2104.13188)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/MichaelFan01/STDC-Seg">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.20.0/mmseg/models/backbones/stdc.py#L394">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

BiSeNet has been proved to be a popular two-stream network for real-time segmentation. However, its principle of adding an extra path to encode spatial information is time-consuming, and the backbones borrowed from pretrained tasks, e.g., image classification, may be inefficient for image segmentation due to the deficiency of task-specific design. To handle these problems, we propose a novel and efficient structure named Short-Term Dense Concatenate network (STDC network) by removing structure redundancy. Specifically, we gradually reduce the dimension of feature maps and use the aggregation of them for image representation, which forms the basic module of STDC network. In the decoder, we propose a Detail Aggregation module by integrating the learning of spatial information into low-level layers in single-stream manner. Finally, the low-level features and deep features are fused to predict the final segmentation results. Extensive experiments on Cityscapes and CamVid dataset demonstrate the effectiveness of our method by achieving promising trade-off between segmentation accuracy and inference speed. On Cityscapes, we achieve 71.9% mIoU on the test set with a speed of 250.4 FPS on NVIDIA GTX 1080Ti, which is 45.2% faster than the latest methods, and achieve 76.8% mIoU with 97.0 FPS while inferring on higher resolution images.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/143640374-d0709587-edb2-4821-bb60-340035f6ad8f.png" width="60%"/>
</div>

## Usage

We have provided [ImageNet Pretrained STDCNet Weights](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1) models converted from [official repo](https://github.com/MichaelFan01/STDC-Seg).

If you want to convert keys on your own to use official repositories' pre-trained models, we also provide a script [`stdc2mmseg.py`](../../tools/model_converters/stdc2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/MichaelFan01/STDC-Seg) to MMSegmentation style.

```shell
python tools/model_converters/stdc2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH} ${STDC_TYPE}
```

E.g.

```shell
python tools/model_converters/stdc2mmseg.py ./STDCNet813M_73.91.tar ./pretrained/stdc1.pth STDC1

python tools/model_converters/stdc2mmseg.py ./STDCNet1446_76.47.tar ./pretrained/stdc2.pth STDC2
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### Cityscapes

| Method | Backbone             | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                        | download                                                                                                                                                                                                                                                                                                                                             |
| ------ | -------------------- | --------- | ------: | -------- | -------------- | ------ | ----: | ------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| STDC   | STDC1 (No Pretrain)  | 512x1024  |   80000 | 7.15     | 23.06          | V100   | 71.82 | 73.89         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/stdc/stdc1_4xb12-80k_cityscapes-512x1024.py)          | [model](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_512x1024_80k_cityscapes/stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_512x1024_80k_cityscapes/stdc1_512x1024_80k_cityscapes_20220224_073048.log.json)                                     |
| STDC   | STDC1                | 512x1024  |   80000 | -        | -              | V100   | 74.94 | 76.97         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/stdc/stdc1_in1k-pre_4xb12-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_in1k-pre_512x1024_80k_cityscapes/stdc1_in1k-pre_512x1024_80k_cityscapes_20220224_141648-3d4c2981.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_in1k-pre_512x1024_80k_cityscapes/stdc1_in1k-pre_512x1024_80k_cityscapes_20220224_141648.log.json) |
| STDC   | STDC2  (No Pretrain) | 512x1024  |   80000 | 8.27     | 23.71          | V100   | 73.15 | 76.13         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/stdc/stdc2_4xb12-80k_cityscapes-512x1024.py)          | [model](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc2_512x1024_80k_cityscapes/stdc2_512x1024_80k_cityscapes_20220222_132015-fb1e3a1a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc2_512x1024_80k_cityscapes/stdc2_512x1024_80k_cityscapes_20220222_132015.log.json)                                     |
| STDC   | STDC2                | 512x1024  |   80000 | -        | -              | V100   | 76.67 | 78.67         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/stdc/stdc2_in1k-pre_4xb12-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc2_in1k-pre_512x1024_80k_cityscapes/stdc2_in1k-pre_512x1024_80k_cityscapes_20220224_073048-1f8f0f6c.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc2_in1k-pre_512x1024_80k_cityscapes/stdc2_in1k-pre_512x1024_80k_cityscapes_20220224_073048.log.json) |

Note:

- For STDC on Cityscapes dataset, default setting is 4 GPUs with 12 samples per GPU in training.
- `No Pretrain` means the model is trained from scratch.
- The FPS is for reference only. The environment is also different from paper setting, whose input size is `512x1024` and `768x1536`, i.e., 50% and 75% of our input size, respectively and using TensorRT.
- The parameter `fusion_kernel` in `STDCHead` is not learnable. In official repo, `find_unused_parameters=True` is set [here](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L220). You may check it by printing model parameters of original repo on your own.

## Citation

```bibtex
@inproceedings{fan2021rethinking,
  title={Rethinking BiSeNet For Real-time Semantic Segmentation},
  author={Fan, Mingyuan and Lai, Shenqi and Huang, Junshi and Wei, Xiaoming and Chai, Zhenhua and Luo, Junfeng and Wei, Xiaolin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9716--9725},
  year={2021}
}
```
