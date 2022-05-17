# Tutorial 3: Customize Data Pipelines

## Design of Data pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading
with multiple workers. `Dataset` returns a dict of data items corresponding
the arguments of models' forward method.
Since the data in semantic segmentation may not be the same size,
we introduce a new `DataContainer` type in MMCV to help collect and distribute
data of different size.
See [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) for more details.

The data preparation pipeline and the dataset is decomposed. Usually a dataset
defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.
A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

The operations are categorized into data loading, pre-processing, formatting and test-time augmentation.

Here is an pipeline example for PSPNet.

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

For each operation, we list the related dict fields that are added/updated/removed.

### Data loading

`LoadImageFromFile`

- add: img, img_shape, ori_shape

`LoadAnnotations`

- add: gt_semantic_seg, seg_fields

### Pre-processing

`Resize`

- add: scale, scale_idx, pad_shape, scale_factor, keep_ratio
- update: img, img_shape, \*seg_fields

`RandomFlip`

- add: flip
- update: img, \*seg_fields

`Pad`

- add: pad_fixed_size, pad_size_divisor
- update: img, pad_shape, \*seg_fields

`RandomCrop`

- update: img, pad_shape, \*seg_fields

`Normalize`

- add: img_norm_cfg
- update: img

`SegRescale`

- update: gt_semantic_seg

`PhotoMetricDistortion`

- update: img

### Formatting

`ToTensor`

- update: specified by `keys`.

`ImageToTensor`

- update: specified by `keys`.

`Transpose`

- update: specified by `keys`.

`ToDataContainer`

- update: specified by `fields`.

`DefaultFormatBundle`

- update: img, gt_semantic_seg

`Collect`

- add: img_meta (the keys of img_meta is specified by `meta_keys`)
- remove: all other keys except for those specified by `keys`

### Test time augmentation

`MultiScaleFlipAug`

## Extend and use custom pipelines

1. Write a new pipeline in any file, e.g., `my_pipeline.py`. It takes a dict as input and return a dict.

   ```python
   from mmseg.datasets import PIPELINES

   @PIPELINES.register_module()
   class MyTransform:

       def __call__(self, results):
           results['dummy'] = True
           return results
   ```

2. Import the new class.

   ```python
   from .my_pipeline import MyTransform
   ```

3. Use it in config files.

   ```python
   img_norm_cfg = dict(
       mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
   crop_size = (512, 1024)
   train_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='LoadAnnotations'),
       dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
       dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
       dict(type='RandomFlip', flip_ratio=0.5),
       dict(type='PhotoMetricDistortion'),
       dict(type='Normalize', **img_norm_cfg),
       dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
       dict(type='MyTransform'),
       dict(type='DefaultFormatBundle'),
       dict(type='Collect', keys=['img', 'gt_semantic_seg']),
   ]
   ```
