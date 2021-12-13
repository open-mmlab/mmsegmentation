# Rethinking BiSeNet For Real-time Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/MichaelFan01/STDC-Seg">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.20.0/mmseg/models/backbones/stdc.py#L394">Code Snippet</a>

## Abstract

BiSeNet has been proved to be a popular two-stream network for real-time segmentation. However, its principle of adding an extra path to encode spatial information is time-consuming, and the backbones borrowed from pretrained tasks, e.g., image classification, may be inefficient for image segmentation due to the deficiency of task-specific design. To handle these problems, we propose a novel and efficient structure named Short-Term Dense Concatenate network (STDC network) by removing structure redundancy. Specifically, we gradually reduce the dimension of feature maps and use the aggregation of them for image representation, which forms the basic module of STDC network. In the decoder, we propose a Detail Aggregation module by integrating the learning of spatial information into low-level layers in single-stream manner. Finally, the low-level features and deep features are fused to predict the final segmentation results. Extensive experiments on Cityscapes and CamVid dataset demonstrate the effectiveness of our method by achieving promising trade-off between segmentation accuracy and inference speed. On Cityscapes, we achieve 71.9% mIoU on the test set with a speed of 250.4 FPS on NVIDIA GTX 1080Ti, which is 45.2% faster than the latest methods, and achieve 76.8% mIoU with 97.0 FPS while inferring on higher resolution images.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/143640374-d0709587-edb2-4821-bb60-340035f6ad8f.png" width="60%"/>
</div>

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.13188">STDC (CVPR'2021)</a></summary>

```latex
@inproceedings{fan2021rethinking,
  title={Rethinking BiSeNet For Real-time Semantic Segmentation},
  author={Fan, Mingyuan and Lai, Shenqi and Huang, Junshi and Wei, Xiaoming and Chai, Zhenhua and Luo, Junfeng and Wei, Xiaolin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9716--9725},
  year={2021}
}
```

</details>

## Usage

To use original repositories' [ImageNet Pretrained STDCNet Weights](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1) , it is necessary to convert keys.

We provide a script [`stdc2mmseg.py`](../../tools/model_converters/stdc2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/MichaelFan01/STDC-Seg) to MMSegmentation style.

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

| Method    | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                  | download                                                                                                                                                                                                                                                       |
| --------- | --------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| STDC1 (No Pretrain) | STDC1 | 512x1024  | 80000 | 7.15 | 23.06 | 71.52 | 73.35 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/stdc/stdc1_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/v0.5/stdc/stdc1_512x1024_80k_cityscapes/stdc1_512x1024_80k_cityscapes_20211125_211245-2c8ba4c5.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_512x1024_80k_cityscapes/stdc1_512x1024_80k_cityscapes_20211125_211245.log.json) |
| STDC1| STDC1 | 512x1024  | 80000 | - | - | 75.10 | 77.72 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/stdc/stdc1_in1k-pre_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_in1k-pre_512x1024_80k_cityscapes/stdc1_in1k-pre_512x1024_80k_cityscapes_20211125_213942-880bb7d0.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_in1k-pre_512x1024_80k_cityscapes/stdc1_in1k-pre_512x1024_80k_cityscapes_20211125_213942.log.json) |
| STDC2 (No Pretrain) | STDC2 | 512x1024  | 80000 | 8.27 | 23.71 | 73.20 | 75.55 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/stdc/stdc2_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc2_512x1024_80k_cityscapes/stdc2_512x1024_80k_cityscapes_20211125_222450-82333ae0.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc2_512x1024_80k_cityscapes/stdc2_512x1024_80k_cityscapes_20211125_222450.log.json) |
| STDC2 | STDC2 | 512x1024  | 80000 | - | - | 77.17 | 79.01 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/stdc/stdc2_in1k-pre_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc2_in1k-pre_512x1024_80k_cityscapes/stdc2_in1k-pre_512x1024_80k_cityscapes_20211125_220437-d2c469f8.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc2_in1k-pre_512x1024_80k_cityscapes/stdc2_in1k-pre_512x1024_80k_cityscapes_20211125_220437.log.json) |

Note:

- For STDC on Cityscapes dataset, default setting is 4 GPUs with 12 samples per GPU in training.
- `No Pretrain` means the model is trained from scratch.
- The FPS is for reference only. The environment is also different from paper setting, whose input size is `512x1024` and `768x1536`, i.e., 50% and 75% of our input size, respectively and using TensorRT.
- The parameter `fusion_kernel` in `STDCHead` is not learnable. In official repo, `find_unused_parameters=True` is set [here](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L220). You may check it by printing model parameters of original repo on your own.
