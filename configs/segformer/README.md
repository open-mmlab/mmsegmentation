# SegFormer

> [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/NVlabs/SegFormer">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/mit.py#L246">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perception (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding, thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from different layers, and thus combining both local attention and global attention to render powerful representations. We show that this simple and lightweight design is the key to efficient segmentation on Transformers. We scale our approach up to obtain a series of models from SegFormer-B0 to SegFormer-B5, reaching significantly better performance and efficiency than previous counterparts. For example, SegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters, being 5x smaller and 2.2% better than the previous best method. Our best model, SegFormer-B5, achieves 84.0% mIoU on Cityscapes validation set and shows excellent zero-shot robustness on Cityscapes-C. Code will be released at: [this http URL](https://github.com/NVlabs/SegFormer).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142902600-e188073e-5744-4ba9-8dbf-9316e55c74aa.png" width="70%"/>
</div>

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`mit2mmseg.py`](../../tools/model_converters/mit2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/NVlabs/SegFormer) to MMSegmentation style.

```shell
python tools/model_converters/mit2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20K

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device   |  mIoU | mIoU(ms+flip) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                               |
| --------- | -------- | --------- | ------: | -------: | -------------- | -------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Segformer | MIT-B0   | 512x512   |  160000 |      2.1 | 51.32          | 1080 Ti  | 37.41 | 38.34         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530.log.json) |
| Segformer | MIT-B1   | 512x512   |  160000 |      2.6 | 47.66          | TITAN Xp | 40.97 | 42.54         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106.log.json) |
| Segformer | MIT-B2   | 512x512   |  160000 |      3.6 | 30.88          | TITAN Xp | 45.58 | 47.03         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103.log.json) |
| Segformer | MIT-B3   | 512x512   |  160000 |      4.8 | 22.11          | V100     | 47.82 | 48.81         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b3_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410.log.json) |
| Segformer | MIT-B4   | 512x512   |  160000 |      6.1 | 15.45          | V100     | 48.46 | 49.76         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b4_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_512x512_160k_ade20k/segformer_mit-b4_512x512_160k_ade20k_20210728_183055-7f509d7d.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_512x512_160k_ade20k/segformer_mit-b4_512x512_160k_ade20k_20210728_183055.log.json) |
| Segformer | MIT-B5   | 512x512   |  160000 |      7.2 | 11.89          | V100     | 49.13 | 50.22         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235.log.json) |
| Segformer | MIT-B5   | 640x640   |  160000 |     11.5 | 11.30          | V100     | 49.62 | 50.36         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243.log.json) |

Evaluation with AlignedResize:

| Method    | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) |
| --------- | -------- | --------- | ------: | ----: | ------------- |
| Segformer | MIT-B0   | 512x512   |  160000 |  38.1 | 38.57         |
| Segformer | MIT-B1   | 512x512   |  160000 | 41.64 | 42.76         |
| Segformer | MIT-B2   | 512x512   |  160000 | 46.53 | 47.49         |
| Segformer | MIT-B3   | 512x512   |  160000 | 48.46 | 49.14         |
| Segformer | MIT-B4   | 512x512   |  160000 | 49.34 | 50.29         |
| Segformer | MIT-B5   | 512x512   |  160000 | 50.08 | 50.72         |
| Segformer | MIT-B5   | 640x640   |  160000 | 50.58 | 50.8          |

We replace `AlignedResize` in original implementatiuon to `Resize + ResizeToMultiple`. If you want to test by
using `AlignedResize`, you can change the dataset pipeline like this:

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # resize image to multiple of 32, improve SegFormer by 0.5-1.0 mIoU.
    dict(type='ResizeToMultiple', size_divisor=32),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
```

### Cityscapes

The lower fps result is caused by the sliding window inference scheme (window size:1024x1024).

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                                | download                                                                                                                                                                                                                                                                                                                                                                                       |
| --------- | -------- | --------- | ------: | -------: | -------------- | ------ | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Segformer | MIT-B0   | 1024x1024 |  160000 |     3.64 | 4.74           | V100   | 76.54 | 78.22         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857.log.json) |
| Segformer | MIT-B1   | 1024x1024 |  160000 |     4.49 | 4.3            | V100   | 78.56 | 79.73         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213.log.json) |
| Segformer | MIT-B2   | 1024x1024 |  160000 |     7.42 | 3.36           | V100   | 81.08 | 82.18         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205.log.json) |
| Segformer | MIT-B3   | 1024x1024 |  160000 |    10.86 | 2.53           | V100   | 81.94 | 83.14         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823.log.json) |
| Segformer | MIT-B4   | 1024x1024 |  160000 |    15.07 | 1.88           | V100   | 81.89 | 83.38         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709.log.json) |
| Segformer | MIT-B5   | 1024x1024 |  160000 |    18.00 | 1.39           | V100   | 82.25 | 83.48         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934.log.json) |

## Citation

```bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```
