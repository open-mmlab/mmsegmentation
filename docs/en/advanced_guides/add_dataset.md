# Add New Datasets

## Data configuration

`train_dataloader`, `val_dataloader` and `test_dataloader` in config file are the variables for data configuration, to define the arguments that are used in datasets and dataloaders.

Here is an example of dataloader configuration:

```python
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
```

- The setting of `xxx_dataloader` works for model training, validation and testing that are by [`build and registry`](https://github.com/open-mmlab/mmengine/blob/master/docs/en/tutorials/registry.md) mechanism.

- `batch_size`: the number of a batch sample of each gpu, and the total batch sizes of training is equal to `batch_size` times gpu number, e.g. when using 8 gpus for distributed data parallel trainig and `batch_size=4`, the total batch sizes is `8*4=32`.

- `num_workers`: How many subprocesses per gpu to use for data loading. `0` means that the data will be loaded in the main process.

**Note:** Default setting of `batch_size` is `1` in MMSegmentation when model testing and validation due to different shapes of samples in a dataset.

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
dataset_A_train_dataloader = dict(
    # There is a configuration for dataset.
    dataset=dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
            )
        )
    )
```

### Concatenate dataset

If the datasets you want to concatenate are in the same type with different annotation files, you can concatenate the dataset configs like the following.

````
You may concatenate two `dataset`.

  ```python
    dataset_train_1 = dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir_1'),
        ann_file='ann_file_1.txt',
        pipeline=train_pipeline)
    dataset_train_2 = dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir_2'),
        ann_file='ann_file_2.txt',
        pipeline=train_pipeline)
    train_dataloader = dict(
        batch_size=4,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='InfiniteSampler', shuffle=True),
        dataset=dict(type='ConcatDataset', datasets=[dataset_train_1, dataset_train_2]))
  ```

  In this case, `ann_dir_1` and `ann_dir_2` are corresponding to `ann_file_1.txt` and `ann_file_2.txt`.
````

### Multi-image Mix Dataset

We use `MultiImageMixDataset` as a wrapper to mix images from multiple datasets.
`MultiImageMixDataset` can be used by multiple images mixed data augmentation
like mosaic and mixup.

An example of using `MultiImageMixDataset` with `Mosaic` data augmentation:

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomMosaic', prob=1),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            classes=classes,
            palette=palette,
            type=dataset_type,
            reduce_zero_label=False,
            data_prefix=dict(img_path=data_root + "images/train",
                             seg_map_path=data_root + "annotations/train"),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
            ]
        ),
        pipeline=train_pipeline)
)
```

## How to prepare your own dataset

Preparation of your own dataset is divided into following steps:

- Dataset preparation, generating correct data format and saving it to correct place like `./data/`.
- Define your dataset in `./mmseg/datasets/` to register it in `DATASETS`.
- Setting hyperparameters of training and validation in `./configs/_base_/datasets/`, such as data augmentation strategy, image crop size and image pixel normalization.

The work directory should follow the structure below.

```none
mmsegmentation
   |
   |- data
   |     |- my_dataset                 # your dataset after data conversion
   |- mmseg
   |     |- datasets
   |     |     |- __init__.py          # import your dataset class here
   |     |     |- my_dataset.py               ## implement your dataset class
   |     |     |- ...
   |- configs
   |     |- _base_
   |     |     |- datasets
   |     |     |     |- my_dataset_config.py      # config file of your dataset
   |     |     |- ...
   |     |- ...
   |- ...
```

### Preparation of dataset

First, convert your dataset into following format, where `img_suffix` and `seg_map_suffix` are usually `.png` and `.jpg`, they are set in each dataset class like [here](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/datasets/ade.py#L85-L86).

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

After your dataset conversion done above, create a new file `my_dataset.py` in `./mmseg/dataset/`, where defining `MyDataset` class could be registered into `DATASETS` of MMEngine and be used by model.

```python
from .builder import DATASETS
from .custom import CustomDataset

# Register MyDataset class into DATASETS
@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    METAINFO = dict(
        # Class names of your dataset annotations, i.e., actual names of corresponding label 0, 1, 2 in annotation segmentation maps
        classes = ('background', 'label_a', 'label_b', 'label_c',
           'label_d', ...)
        # BGR value of corresponding classes, which are used for visualization
        palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
           [255, 255, 0], ...]
    )

    # The formats of image and segmentation map are both .png in this case
    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False, # When reduce_zero_label is False label 0 (first one in CLASSES above) would be calculated in loss.
            **kwargs)
```

`classes` and `palette` in `METAINFO` are defined in `./mmseg/dataset/my_dataset.py`, which are class names and BGR values of annotations, respectively.
Besides,`palette` would only be used in prediction visulization, which would not affect process of training and validation.

Specifically, you could set `reduce_zero_label=True` when you want to reset label `0` in annotation files to `255` which would be ignored when calculating loss, and all other label `i` would be `i-1`. More details could be found [here](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/datasets/transforms/loading.py#L110-L114).

After creating `./mmseg/dataset/my_dataset.py`，it is also necessary to add it in `./mmseg/dataset/__init__.py`:

```python
from .my_dataset import MyDataset

__all__ = [
    ...,
    'MyDataset'
]
```

### Set config file of dataset

After defining your dataset, it is also necessary to define configs of your dataset `my_dataset_config.py` in `./configs/_base_/datasets/`, which would be used concurrently with other config parameters in training and inference.

```python
# Class name defined in ./mmseg/datasets/my_dataset.py
dataset_type = 'MyDataset'
# Correct path of your dataset
data_root = 'data/my_dataset'

crop_size = (512, 512) # Crop size of image in training
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

# Training dataset config
train_dataloader = dict(
    batch_size=4,  # Batch size of a single GPU
    num_workers=4, # Worker to pre-fetch data for each single GPU
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root, # The root of dataset.
        data_prefix=dict(
            # The image directory of dataset.
            img_path='images/training',
            # The annotation directory of dataset.
            seg_map_path='annotations/training'),
        # pipeline, this is passed by the train_pipeline created before.
        pipeline=train_pipeline))
# Validation dataset config
val_dataloader = dict(
    ...
)
# Test dataset config
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
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
