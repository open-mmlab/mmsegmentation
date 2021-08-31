# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`mit2mmseg.py`](../../tools/model_converters/mit2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/NVlabs/SegFormer) to MMSegmentation style.

```shell
python tools/model_converters/swin2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

## Results and models

### ADE20k

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ------: | -------: | -------------- | ---: | ------------- | ------ | -------- |
|Segformer | MIT-B0 | 512x512 | 160000 | 2.1 | 51.32 | 37.41 | 38.34 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530.log.json) |
|Segformer | MIT-B1 | 512x512 | 160000 | 2.6 | 47.66 | 40.97 | 42.54 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segformer/segformer_mit-b1_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106.log.json) |
|Segformer | MIT-B2 | 512x512 | 160000 | 3.6 | 30.88 | 45.58 | 47.03 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103.log.json) |
|Segformer | MIT-B3 | 512x512 | 160000 | 4.8 | 22.11 | 47.82 | 48.81 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segformer/segformer_mit-b3_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410.log.json) |
|Segformer | MIT-B4 | 512x512 | 160000 | 6.1 | 15.45 | 48.46 | 49.76 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segformer/segformer_mit-b4_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_512x512_160k_ade20k/segformer_mit-b4_512x512_160k_ade20k_20210728_183055-7f509d7d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_512x512_160k_ade20k/segformer_mit-b4_512x512_160k_ade20k_20210728_183055.log.json) |
|Segformer | MIT-B5 | 512x512 | 160000 | 7.2 | 11.89 | 49.13 | 50.22 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235.log.json) |
|Segformer | MIT-B5 | 640x640 | 160000 | 11.5 | 11.30 | 49.62 | 50.36 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segformer/segformer_mit-b5_640x640_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243.log.json) |

Evaluation with AlignedResize:

  | Method | Backbone | Crop Size | Lr schd | mIoU | mIoU(ms+flip) |
  | ------ | -------- | --------- | ------: | ---: | ------------- |
  |Segformer | MIT-B0 | 512x512 | 160000 | 38.1  | 38.57 |
  |Segformer | MIT-B1 | 512x512 | 160000 | 41.64 | 42.76 |
  |Segformer | MIT-B2 | 512x512 | 160000 | 46.53 | 47.49 |
  |Segformer | MIT-B3 | 512x512 | 160000 | 48.46 | 49.14 |
  |Segformer | MIT-B4 | 512x512 | 160000 | 49.34 | 50.29 |
  |Segformer | MIT-B5 | 512x512 | 160000 | 50.08 | 50.72 |
  |Segformer | MIT-B5 | 640x640 | 160000 | 50.58 | 50.8  |

We replace `AlignedResize` in original implementatiuon to `Resize + ResizeToMultiple`. If you want to test by
using `AlignedResize`, you can change the dataset pipeline like this:

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # resize image to multiple of 32, improve SegFormer by 0.5-1.0 mIoU.
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```
