# Data Transforms

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
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
```

For each operation, we list the related dict fields that are added/updated/removed.
Before pipelines, the information we can directly obtain from the datasets are img_path, seg_map_path.

### Data loading

`LoadImageFromFile`

- add: img, img_shape, ori_shape

`LoadAnnotations`

- add: seg_fields, gt_seg_map

### Pre-processing

`RandomResize`

- add: scale, scale_factor, keep_ratio
- update: img, img_shape, gt_seg_map

`Resize`

- add: scale, scale_factor, keep_ratio
- update: img, gt_seg_map, img_shape

`RandomCrop`

- update: img, pad_shape, gt_seg_map

`RandomFlip`

- add: flip, flip_direction
- update: img, gt_seg_map

`PhotoMetricDistortion`

- update: img

### Formatting

`PackSegInputs`

- add: inputs, data_sample
- remove: keys specified by `meta_keys` (merged into the metainfo of data_sample), all other keys
