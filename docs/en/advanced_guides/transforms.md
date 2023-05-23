# Data Transforms

In this tutorial, we introduce the design of transforms pipeline in MMSegmentation.

The structure of this guide is as follows:

- [Data Transforms](#data-transforms)
  - [Design of Data pipelines](#design-of-data-pipelines)
    - [Data loading](#data-loading)
    - [Pre-processing](#pre-processing)
    - [Formatting](#formatting)

## Design of Data pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading with multiple workers. `Dataset` returns a dict of data items corresponding the arguments of models' forward method. Since the data in semantic segmentation may not be the same size, we introduce a new `DataContainer` type in MMCV to help collect and distribute data of different size. See [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) for more details.

In 1.x version of MMSegmentation, all data transformations are inherited from [`BaseTransform`](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/transforms/base.py#L6).

The input and output types of transformations are both dict. A simple example is as follows:

```python
>>> from mmseg.datasets.transforms import LoadAnnotations
>>> transforms = LoadAnnotations()
>>> img_path = './data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png.png'
>>> gt_path = './data/cityscapes/gtFine/train/aachen/aachen_000015_000019_gtFine_instanceTrainIds.png'
>>> results = dict(
>>>     img_path=img_path,
>>>     seg_map_path=gt_path,
>>>     reduce_zero_label=False,
>>>     seg_fields=[])
>>> data_dict = transforms(results)
>>> print(data_dict.keys())
dict_keys(['img_path', 'seg_map_path', 'reduce_zero_label', 'seg_fields', 'gt_seg_map'])
```

The data preparation pipeline and the dataset are decomposed. Usually a dataset defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict. A pipeline consists of a sequence of operations. Each operation takes a dict as input and also outputs a dict for the next transform.

The operations are categorized into data loading, pre-processing, formatting and test-time augmentation.

Here is a pipeline example for PSPNet:

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
    # does not need to resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
```

For each operation, we list the related dict fields that are `added`/`updated`/`removed`. Before pipelines, the information we can directly obtain from the datasets are `img_path` and `seg_map_path`.

### Data loading

`LoadImageFromFile`: Load an image from file.

- add: `img`, `img_shape`, `ori_shape`

`LoadAnnotations`: Load semantic segmentation maps provided by dataset.

- add: `seg_fields`, `gt_seg_map`

### Pre-processing

`RandomResize`: Random resize image & segmentation map.

- add: `scale`, `scale_factor`, `keep_ratio`
- update: `img`, `img_shape`, `gt_seg_map`

`Resize`: Resize image & segmentation map.

- add: `scale`, `scale_factor`, `keep_ratio`
- update: `img`, `gt_seg_map`, `img_shape`

`RandomCrop`: Random crop image & segmentation map.

- update: `img`, `gt_seg_map`, `img_shape`

`RandomFlip`: Flip the image & segmentation map.

- add: `flip`, `flip_direction`
- update: `img`, `gt_seg_map`

`PhotoMetricDistortion`: Apply photometric distortion to image sequentially, every transformation is applied with a probability of 0.5. The position of random contrast is in second or second to last(mode 0 or 1 below, respectively).

```
1. random brightness
2. random contrast (mode 0)
3. convert color from BGR to HSV
4. random saturation
5. random hue
6. convert color from HSV to BGR
7. random contrast (mode 1)
```

- update: `img`

### Formatting

`PackSegInputs`: Pack the inputs data for the semantic segmentation.

- add: `inputs`, `data_sample`
- remove: keys specified by `meta_keys` (merged into the metainfo of data_sample), all other keys
