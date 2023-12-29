# Add New Datasets

## Add new custom dataset

Here we show how to develop a new custom dataset.

1. Create a new file `mmseg/datasets/example.py`

   ```python
   from mmseg.registry import DATASETS
   from .basesegdataset import BaseSegDataset


   @DATASETS.register_module()
   class ExampleDataset(BaseSegDataset):

       METAINFO = dict(
           classes=('xxx', 'xxx', ...),
           palette=[[x, x, x], [x, x, x], ...])

       def __init__(self, arg1, arg2):
           pass
   ```

2. Import the module in `mmseg/datasets/__init__.py`

   ```python
   from .example import ExampleDataset
   ```

3. Use it by creating a new new dataset config file `configs/_base_/datasets/example_dataset.py`

   ```python
   dataset_type = 'ExampleDataset'
   data_root = 'data/example/'
   ...
   ```

4. Add dataset meta information in `mmseg/utils/class_names.py`

   ```python
   def example_classes():
       return [
           'xxx', 'xxx',
           ...
       ]

   def example_palette():
       return [
           [x, x, x], [x, x, x],
           ...
       ]
   dataset_aliases ={
       'example': ['example', ...],
       ...
   }
   ```

**Note:** If the new dataset does not satisfy the mmseg requirements, a data preprocessing script needs to be prepared in `tools/dataset_converters/`

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

Some datasets don't release the test set or don't release the ground truth of the test set, and we cannot evaluate models locally without the ground truth of the test set, so we set the validation set as the default test set in config files.

About how to build your own datasets or implement a new dataset class please refer to the [datasets guide](./datasets.md) for more detailed information.

**Note:** The annotations are images of shape (H, W), the value pixel should fall in range `[0, num_classes - 1]`.
You may use `'P'` mode of [pillow](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#palette) to create your annotation image with color.

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

In case the dataset you want to concatenate is different, you can concatenate the dataset configs like the following.

```python
dataset_A_train = dict()
dataset_B_train = dict()
concatenate_dataset = dict(
    type='ConcatDataset',
    datasets=[dataset_A_train, dataset_B_train])
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
train_dataloader = dict(
    dataset=dict(
        type='ConcatDataset',
        datasets=[dataset_A_train, dataset_B_train]))

val_dataloader = dict(dataset=dataset_A_val)
test_dataloader = dict(dataset=dataset_A_test)

```

You can refer base dataset [tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html) from mmengine for more details

### Multi-image Mix Dataset

We use `MultiImageMixDataset` as a wrapper to mix images from multiple datasets.
`MultiImageMixDataset` can be used by multiple images mixed data augmentation like mosaic and mixup.

An example of using `MultiImageMixDataset` with `Mosaic` data augmentation:

```python
train_pipeline = [
    dict(type='RandomMosaic', prob=1),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
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
