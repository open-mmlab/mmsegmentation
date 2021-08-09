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

## Results and models

### ADE20k

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ------: | -------: | -------------- | ---: | ------------- | ------ | -------- |
|Segformer | MIT-B0 | 512x512 | 160000 | - | - | 37.41 | 38.34 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B1 | 512x512 | 160000 | - | - | 40.97 | 42.54 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B2 | 512x512 | 160000 | - | - | 45.58 | 47.03 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B3 | 512x512 | 160000 | - | - | 47.82 | 48.81 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B4 | 512x512 | 160000 | - | - | 48.46 | 49.76 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B5 | 512x512 | 160000 | - | - | 49.13 | 50.22 | [config]() | [model]() &#124; [log]() |
|Segformer | MIT-B5 | 512x512 | 160000 | - | - | 49.62 | - | [config]() | [model]() &#124; [log]() |

Evaluation with AlignedResize:

| Method | Backbone | Crop Size | Lr schd | mIoU | mIoU(ms+flip) |
| ------ | -------- | --------- | ------: | ---: | ------------- |
|Segformer | MIT-B0 | 512x512 | 160000 | 38.1  | 38.57 |
|Segformer | MIT-B1 | 512x512 | 160000 | 41.64 | 42.76 |
|Segformer | MIT-B2 | 512x512 | 160000 | 46.53 | 47.49 |
|Segformer | MIT-B3 | 512x512 | 160000 | 48.46 | 49.14 |
|Segformer | MIT-B4 | 512x512 | 160000 | 49.34 | 50.29 |
|Segformer | MIT-B5 | 512x512 | 160000 | 50.08 | 50.72 |
|Segformer | MIT-B5 | 640x640 | 160000 | 50.58 | - |

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
            # resize image to multiple of 32, improve SegFormer by xx mIoU.
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```
