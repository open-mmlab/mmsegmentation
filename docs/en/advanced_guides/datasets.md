# Dataset

In this document, we will introduce functions of `BaseSegDataset` class in MMSegmentation.

## BaseSegDataset

`BaseSegDataset` is a custom dataset class for semantic segmentation task, which is inherited from `BaseDataset` in MMEngine.
More details of `BaseDataset` could be found in [MMEngine BaseDataset Documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/basedataset.md).

All datasets are inherited from `BaseSegDataset` in the 1.x version of MMSegmentation. Some frequently used methods are defined in `BaseSegDataset`,
such as getting meta and raw data information of dataset, updating label mapping and palette information when reducing label `0` (which is usually background) and so on.

`BaseSegDataset` is designed for supervised semantic segmentation task thus both images and annotations are necessary.
The directory structure of corresponding dataset is as below.

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

The image and ground truth annotation pair of `BaseSegDataset` should be of the same except suffix.
A valid image and ground truth annotation filename pair should be like `xxx{img_suffix}` and `xxx{seg_map_suffix}` (extension is also included
in the suffix). Two keys `data_list` and `metainfo` are introduced in `BaseSegDataset` to save raw data (such as image path)
and meta data (such as class names of ground truth) information, respectively.

Next, we would introduce four important parts of `BaseSegDataset`: (1) initialization (2) loading meta information (3) loading raw data information and (4) `__getitem__` method.
More information about `BaseSegDataset` could be found in [basesegdataset.py](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/datasets/basesegdataset.py).

### Initialization

Because of `BaseSegDataset` is inherited from [`BaseDataset`](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/basedataset.md)
in MMEngine, it also follows similar parts of initialization process of `BaseDataset`.

Below is a table of some important methods in `BaseSegDataset` initialization method, which would be executed in order from top to bottom.

| Methods executed in `BaseSegDataset` initialization |                            Usage                            |
| :-------------------------------------------------: | :---------------------------------------------------------: |
|                  `self._metainfo`                   |              Set meta information of dataset.               |
|                `self._join_prefix()`                |            Join paths of images and annotations.            |
|                   `self.pipeline`                   | Build data pipeline for data preprocessing and preparation. |
|                 `self.full_init()`                  |                Full initialize the dataset.                 |

In `BaseSegDataset`, the argument `lazy_init` (Defaults to False) is used to control whether to load annotation during instantiation.
If not using lazy initialization, the `self.full_init()` would be called to execute several methods from top to bottom:

| Methods executed in `self.full_init()` |                           Usage                            |
| :------------------------------------: | :--------------------------------------------------------: |
|        `self.load_data_list()`         |                   Load data information.                   |
|          `self.filter_data()`          | Filter illegal data, such as data that has no annotations. |
|   `self._get_unserialized_subset()`    |           Get subset data according to indices.            |
|        `self._serialize_data()`        |   Serialize `data_list` of corresponding dataset class.    |

In some cases such as visualization, only the meta information of the dataset is necessary, while loading annotation file
is unnecessary. `BaseSegDataset` can skip loading annotations to save time by set `lazy_init=True`, in this case `self.full_init()` would not be executed.

### Loading meta info

Meta information is collected in `self._metainfo` method.
If `metainfo` contains existed filename path, it will be parsed by `list_from_file`, otherwise it is simply parsed as meta information:

```python
def _load_metainfo(cls, metainfo: dict = None) -> dict:
    # avoid `cls.METAINFO` being overwritten by `metainfo`
    cls_metainfo = copy.deepcopy(cls.METAINFO)
    ...
    for k, v in metainfo.items():
        if isinstance(v, str):
            # If type of value is string, and can be loaded from
            # corresponding backend. it means the file name of meta file.
            try:
                cls_metainfo[k] = list_from_file(v)
            except (TypeError, FileNotFoundError):
                warnings.warn(f'{v} is not a meta file, simply parsed as '
                              'meta information')
                cls_metainfo[k] = v
        else:
            cls_metainfo[k] = v
    return cls_metainfo
```

Then `_load_metainfo` returns parsed meta information for train/test process.

### Loading data info

By default, `self.load_data_list()` would be called in `self.full_init()`. In `BaseSegDataset`,
`self.load_data_list()` function is overwritten where annotation path `seg_map_path` would be added from annotation file directory or its meta file.

Moreover, in `BaseSegDataset`, `label_map` and `reduce_zero_label` are also added in meta information dict:

The `label_map` is used to require label mapping from old classes in `cls.METAINFO` to new classes in `self._metainfo`,
which changes pixel labels in `load_data_list`.
`label_map` is a dictionary, whose keys are the old label ids and values are the new label ids.
`label_map` is not `None` if and only if (1) old classes in `cls.METAINFO` is not equal to new classes in `self._metainfo`
and (2) both of `cls.METAINFO` and `self._metainfo` are not `None`.

The `reduce_zero_label`(Default to `False`) controls whether to mark label zero as ignored.
Because in semantic segmentation dataset such as [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k),
label `0` usually stands for background, which is not included in classes list in meta information.
If `reduce_zero_label=True`, [`LoadAnnotations`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/datasets/transforms/loading.py#L107-L118)
data transform would ignore label 0 and reduce all other label value by 1.

### `__getitem__` method

By default, `BaseSegDataset` inherits `__getitem__` method in `BaseDataset`.

In `__getitem__` method, `prepare_data` is called to get the processed data, where data loading pipeline consists of the following steps:

1. Fetch data information by index, implemented by `get_data_info`.
2. Apply data transforms to data, implemented by `pipeline`.

Finally, it returns a dict `data` which includes the idx-th image and data information of dataset after `self.pipeline`.

If you want to implement a new dataset class, you only need to implement `load_data_list` function.
We also encourage users to use the original data loading logic provided by `BaseDataset`.
If the default loading logic is difficult to meet your needs, you can overwrite the `__getitem__` interface to implement your data loading logic.

## Differences between CustomDataset(v0.x)

In MMSegmentation \< 1.x, it uses [`CustomDataset`](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/custom.py#L19) as basic dataset class in MMSegmentation.

`BaseSegDataset` is inherited from `BaseDataset` in MMEngine while `CustomDataset` is inherited from `Dataset` in official PyTorch. The differences between `CustomDataset` and `BaseSegDataset` are lied below:

- `BaseSegDataset` removes all methods about evaluations in `CustomDataset`, such as `format_results`, `pre_eval`, `get_gt_seg_map_by_idx`, `get_gt_seg_maps` and `evaluate`.
- `BaseSegDataset` replaces method `load_annotations` to `load_data_list`.
- `BaseSegDataset` replaces member variable `img_infos` and `split` to `data_list` and `ann_file`, respectively.
- `BaseSegDataset` integrates `CLASSES` and `PALETTE` into two fields of `METAINFO` dict.

## Design your own dataset

You may refer to [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/basedataset.md).
