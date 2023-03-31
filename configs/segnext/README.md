# SegNeXt

> [SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation](https://arxiv.org/abs/2209.08575)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/visual-attention-network/segnext">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/backbones/mscan.py#L328">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

We present SegNeXt, a simple convolutional network architecture for semantic segmentation. Recent transformer-based models have dominated the field of semantic segmentation due to the efficiency of self-attention in encoding spatial information. In this paper, we show that convolutional attention is a more efficient and effective way to encode contextual information than the self-attention mechanism in transformers. By re-examining the characteristics owned by successful segmentation models, we discover several key components leading to the performance improvement of segmentation models. This motivates us to design a novel convolutional attention network that uses cheap convolutional operations. Without bells and whistles, our SegNeXt significantly improves the performance of previous state-of-the-art methods on popular benchmarks, including ADE20K, Cityscapes, COCO-Stuff, Pascal VOC, Pascal Context, and iSAID. Notably, SegNeXt outperforms EfficientNet-L2 w/ NAS-FPN and achieves 90.6% mIoU on the Pascal VOC 2012 test leaderboard using only 1/10 parameters of it. On average, SegNeXt achieves about 2.0% mIoU improvements compared to the state-of-the-art methods on the ADE20K datasets with the same or fewer computations. Code is available at [this https URL](https://github.com/uyzhang/JSeg) (Jittor) and [this https URL](https://github.com/Visual-Attention-Network/SegNeXt) (Pytorch).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/215688018-5d4c8366-7793-4fdf-9397-960a09fac951.png" width="70%"/>
</div>

## Results and models

### ADE20K

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device | mIoU  | mIoU(ms+flip) | config                                                                                                                              | download                                                                                                                                                                                                                                                                                                                                                                                   |
| ------- | -------- | --------- | ------- | -------- | -------------- | ------ | ----- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SegNeXt | MSCAN-T  | 512x512   | 160000  | 17.88    | 52.38          | A100   | 41.50 | 42.59         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-t_1x16_512x512_adamw_160k_ade20k/segnext_mscan-t_1x16_512x512_adamw_160k_ade20k_20230210_140244-05bd8466.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-t_1x16_512x512_adamw_160k_ade20k/segnext_mscan-t_1x16_512x512_adamw_160k_ade20k_20230210_140244.log.json) |
| SegNeXt | MSCAN-S  | 512x512   | 160000  | 21.47    | 42.27          | A100   | 44.16 | 45.81         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segnext/segnext_mscan-s_1xb16-adamw-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-s_1x16_512x512_adamw_160k_ade20k/segnext_mscan-s_1x16_512x512_adamw_160k_ade20k_20230214_113014-43013668.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-s_1x16_512x512_adamw_160k_ade20k/segnext_mscan-s_1x16_512x512_adamw_160k_ade20k_20230214_113014.log.json) |
| SegNeXt | MSCAN-B  | 512x512   | 160000  | 31.03    | 35.15          | A100   | 48.03 | 49.68         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053.log.json) |
| SegNeXt | MSCAN-L  | 512x512   | 160000  | 43.32    | 22.91          | A100   | 50.99 | 52.10         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segnext/segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-l_1x16_512x512_adamw_160k_ade20k/segnext_mscan-l_1x16_512x512_adamw_160k_ade20k_20230209_172055-19b14b63.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-l_1x16_512x512_adamw_160k_ade20k/segnext_mscan-l_1x16_512x512_adamw_160k_ade20k_20230209_172055.log.json) |

Note:

- When we integrated SegNeXt into MMSegmentation, we modified some layers' names to make them more precise and concise without changing the model architecture. Therefore, the keys of pre-trained weights are different from the [original weights](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/), but don't worry about these changes. we have converted them and uploaded the checkpoints, you might find URL of pre-trained checkpoints in config files and can use them directly for training.

- The total batch size is 16. We trained for SegNeXt with a single GPU as the performance degrades significantly when using`SyncBN` (mainly in `OverlapPatchEmbed` modules of `MSCAN`) of PyTorch 1.9.

- There will be subtle differences when model testing as Non-negative Matrix Factorization (NMF) in `LightHamHead` will be initialized randomly. To control this randomness, please set the random seed when model testing. You can modify [`./tools/test.py`](https://github.com/open-mmlab/mmsegmentation/blob/main/tools/test.py) like:

```python
def main():
    from mmengine.runner import seg_random_seed
    random_seed = xxx # set random seed recorded in training log
    set_random_seed(random_seed, deterministic=False)
    ...
```

- This model performance is sensitive to the seed values used, please refer to the log file for the specific settings of the seed. If you choose a different seed, the results might differ from the table results. Take SegNeXt Large for example, its results range from 49.60 to 51.0.

## Citation

```bibtex
@article{guo2022segnext,
  title={SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Hou, Qibin and Liu, Zhengning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2209.08575},
  year={2022}
}
```
