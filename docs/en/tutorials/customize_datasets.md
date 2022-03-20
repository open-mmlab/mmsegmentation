# Tutorial 2: Customize Datasets

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
