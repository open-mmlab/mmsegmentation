# Tutorial 2: Customize Datasets

## How to prepare your own dataset

Preparation of your own dataset is divided into following steps:

- Dataset preparation, generating correct data format and saving it to correct place like `./data/`.
- Define your dataset in `./mmseg/datasets/` to register it in `DATASETS`.
- Setting hyperparameters of training and validation in `./configs/_base_/datasets/`, such as data augmentation strategy, image crop size and image pixel normalization.

Files needed to be modified are listed below:

```none
mmsegmentation
   |
   |- data
   |     |- my_dataset                 # your dataset after data conversion
   |- mmseg
   |     |- datasets
   |     |     |- __init__.py          # add your dataset class here
   |     |     |- my_dataset.py               ## define your dataset class
   |     |     |- ...
   |- configs
   |     |- _base_
   |     |     |- datasets
   |     |     |     |- my_dataset_config.py      # config of your dataset
   |     |     |- ...
   |     |- ...
   |- ...
```

### Preparation of dataset

First, convert your dataset into following format, where `img_suffix` and `seg_map_suffix` are format of images and annotations, which are usually `.png` and `.jpg`.

```none
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   ├── val
│   │   │   │   ├── zzz{img_suffix}
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   ├── val
│   │   │   │   ├── zzz{seg_map_suffix}

```

### Dataset Registration

After your dataset conversion done above, create a new file `my_dataset.py` in `./mmseg/dataset/`, where defining `MyDataset` class could be registered into `DATASETS` of MMCV and be used by model.


```python
from .builder import DATASETS
from .custom import CustomDataset

# Register MyDataset class into DATASETS
@DATASETS.register_module()
class MyDataset(CustomDataset):
    # Class names of your dataset annotations, i.e., actual names of corresponding label 0, 1, 2 in annotation segmentation maps
    CLASSES = ('background', 'label_a', 'label_b', 'label_c',
               'label_d', ...)
    # BGR value of corresponding classes, which are used for visualization
    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], ...]

    # The formats of image and segmentation map are both .png in this case
    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False, # reduce_zero_label is False because label 0 is background (first one in CLASSES above)
            **kwargs)
```

`CLASSES` and `PALETTE` are defined in `./mmseg/dataset/my_dataset.py`, which are class names and BGR values of annotations, respectively.
Besides,`PALETTE` would only be used in prediction visulization, which would not affect process of training and validation.

Specifically, if label 0 in segmentation map is not background, it should be set `reduce_zero_label=True`.

After creating `./mmseg/dataset/my_dataset.py`后，it is also necessary to add it in `./mmseg/dataset/__init__.py`:

```python
# Copyright (c) OpenMMLab. All rights reserved.
from .my_dataset import MyDataset

__all__ = [
    ...,
    'MyDataset'
]
```

### Set config file of dataset

After defining your dataset, it is also necessary to define configs of your dataset `my_dataset_config.py` in `./configs/_base_/datasets/`, which would be used concurrently with other config parameters in training and inference.

```python
# Your dataset type defined in ./mmseg/datasets/__init__.py
dataset_type = 'MyDataset'
# Correct path of your dataset
data_root = 'data/my_dataset'

img_norm_cfg = dict( # This img_norm_cfg is widely used because it is mean and std of ImageNet 1K pretrained model
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (512, 512) # Crop size of image in training
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
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
        img_scale=(512, 512),
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
data = dict(
    samples_per_gpu=4, # Batch size of a single GPU
    workers_per_gpu=4, # Worker to pre-fetch data for each single GPU
    train=dict( # Train dataset config
        type=dataset_type, # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root, # The root of dataset.
        img_dir='images/training', # The image directory of dataset.
        ann_dir='annotations/training',  # The annotation directory of dataset.
        pipeline=train_pipeline), # pipeline, this is passed by the train_pipeline created before.
    val=dict( # Validation dataset config.
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline), # Pipeline is passed by test_pipeline created before.
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))

```
More information of each config parameter could be found [config.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/tutorials/config.md) 。

After dataset preparation is finished, just import `./configs/_base_/datasets/my_dataset_config.py` in model config file, for example:

```python
_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/my_dataset_config.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=YOUR_DATASET_CLASSES), auxiliary_head=dict(num_classes=YOUR_DATASET_CLASSES))

```

By now, you can use your own dataset in MMSegmentation.

## Customize datasets by reorganizing data

The simplest way is to convert your dataset to organize your data into folders.

An example of file structure is as followed.

```none
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val

```

A training pair will consist of the files with same suffix in img_dir/ann_dir.

If `split` argument is given, only part of the files in img_dir/ann_dir will be loaded.
We may specify the prefix of files we would like to be included in the split txt.

More specifically, for a split txt like following,

```none
xxx
zzz
```

Only
`data/my_dataset/img_dir/train/xxx{img_suffix}`,
`data/my_dataset/img_dir/train/zzz{img_suffix}`,
`data/my_dataset/ann_dir/train/xxx{seg_map_suffix}`,
`data/my_dataset/ann_dir/train/zzz{seg_map_suffix}` will be loaded.

:::{note}
The annotations are images of shape (H, W), the value pixel should fall in range `[0, num_classes - 1]`.
You may use `'P'` mode of [pillow](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#palette) to create your annotation image with color.
:::

## Customize datasets by mixing dataset

MMSegmentation also supports to mix dataset for training.
Currently it supports to concat, repeat and multi-image mix datasets.

### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset.
For example, suppose the original dataset is `Dataset_A`, to repeat it, the config looks like the following

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### Concatenate dataset

There 2 ways to concatenate the dataset.

1. If the datasets you want to concatenate are in the same type with different annotation files,
    you can concatenate the dataset configs like the following.

    1. You may concatenate two `ann_dir`.

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = ['anno_dir_1', 'anno_dir_2'],
            pipeline=train_pipeline
        )
        ```

    2. You may concatenate two `split`.

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = 'anno_dir',
            split = ['split_1.txt', 'split_2.txt'],
            pipeline=train_pipeline
        )
        ```

    3. You may concatenate two `ann_dir` and `split` simultaneously.

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = ['anno_dir_1', 'anno_dir_2'],
            split = ['split_1.txt', 'split_2.txt'],
            pipeline=train_pipeline
        )
        ```

        In this case, `ann_dir_1` and `ann_dir_2` are corresponding to `split_1.txt` and `split_2.txt`.

2. In case the dataset you want to concatenate is different, you can concatenate the dataset configs like the following.

    ```python
    dataset_A_train = dict()
    dataset_B_train = dict()

    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train = [
            dataset_A_train,
            dataset_B_train
        ],
        val = dataset_A_val,
        test = dataset_A_test
        )
    ```

A more complex example that repeats `Dataset_A` and `Dataset_B` by N and M times, respectively, and then concatenates the repeated datasets is as the following.

```python
dataset_A_train = dict(
    type='RepeatDataset',
    times=N,
    dataset=dict(
        type='Dataset_A',
        ...
        pipeline=train_pipeline
    )
)
dataset_A_val = dict(
    ...
    pipeline=test_pipeline
)
dataset_A_test = dict(
    ...
    pipeline=test_pipeline
)
dataset_B_train = dict(
    type='RepeatDataset',
    times=M,
    dataset=dict(
        type='Dataset_B',
        ...
        pipeline=train_pipeline
    )
)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_A_train,
        dataset_B_train
    ],
    val = dataset_A_val,
    test = dataset_A_test
)

```

### Multi-image Mix Dataset

We use `MultiImageMixDataset` as a wrapper to mix images from multiple datasets.
`MultiImageMixDataset` can be used by multiple images mixed data augmentation
like mosaic and mixup.

An example of using `MultiImageMixDataset` with `Mosaic` data augmentation:

```python
train_pipeline = [
    dict(type='RandomMosaic', prob=1),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/train",
        ann_dir=data_root + "annotations/train",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
        ]
    ),
    pipeline=train_pipeline
)

```
