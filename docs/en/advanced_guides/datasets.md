# Dataset

In OpenMMLab 2.0, MMEngine implements `BaseDataset` to provides some basic interfaces for downstream OpenMMLab codebase, which would inherit this base dataset class and add customization methods.

The basic function of the MMEngine `BaseDataset` is loading the dataset information which has two categories.
One is meta information, which represents the information related to the dataset itself and sometimes needs to be obtained by the model or other external components.
The other is data information like the path of image and annotation files.

More details of `BaseDataset` could be found in [MMEngine BaseDataset Documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/basedataset.md).

`BaseSegDataset` is a base dataset class for semantic segmentation task in MMSegmentation v1.x, which is inherited from `BaseDataset` in MMEngine.

In this document, we will introduce `BaseSegDataset` class in MMSegmentation.
First, we would introduce standard dataset format in MMSegmentation, while creating a corresponding dataset class is also necessary.
Then, we would introduce some important methods of `BaseSegDataset`:

- (1) initialization of `BaseSegDataset`, including detailed methods which would be called during this process.

- (2) loading meta information and raw data information, the loading function of these two data information categories are basic function of `BaseSegDataset` for training and test process.

- (3) `__getitem__` method used to give index of data information in `BaseSegDataset` initialization, which is similar to `torch.utils.data.Dataset`.

More information about `BaseSegDataset` could be found in [basesegdataset.py](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/datasets/basesegdataset.py).

## Dataset Format

`BaseSegDataset` is designed for supervised semantic segmentation task thus both images and annotations are necessary when model training.
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

The image and ground truth of `BaseSegDataset` should have the same filename except suffix,
so do customization dataset class in MMSegmentation which inherits `BaseSegDataset`.

Take [`CityscapesDataset`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/datasets/cityscapes.py#L7) dataset class for example:

```python
CityscapesDataset.metainfo
# {
#     'classes': ('road', 'sidewalk', 'building', 'wall', ..., 'bicycle')
#     'palette': [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], ..., [119, 11, 32]]
#     'label_map': None
#     'reduce_zero_label': False
# }

CityscapesDataset.get_data_info(0)
# {
#     'img_path': "data/cityscapes/leftImg8bit/xxx/xxx/xxx_leftImg8bit.jpg",
#     'seg_map_path': "data/cityscapes/gtFine/xxx/xxx/xxx_gtFine_labelTrainIds.png",
#     ...
# }

len(CityscapesDataset)
# 2975 when training, 500 when validation

CityscapesDataset[0]
# {
#     'inputs': a tensor with shape (3, H, W), which denotes the value of the image,
#     'data_samples':
#             {
#             'img_path': "data/cityscapes/leftImg8bit/xxx/xxx/xxx_leftImg8bit.jpg",
#             'seg_map_path': "data/cityscapes/gtFine/xxx/xxx/xxx_gtFine_labelTrainIds.png",
#             'img_shape': (H, W),
#             'metainfo': {
#                     ...
#                     }
#             ...
#         }
# }
```

From above we can find A valid image and ground truth annotation filename pair should be like `xxx{img_suffix}`
and `xxx{seg_map_suffix}` (extension is also included in the suffix), `img_suffix='_leftImg8bit.png'` and `seg_map_suffix='_gtFine_labelTrainIds.png'` in `CityscapesDataset`.
`get_data_info` and `metainfo` are also introduced in `BaseSegDataset`(so do `CityscapesDataset`) to save raw data (such as image path)
and meta data (such as class names of ground truth) information, respectively.

Next, we would introduce some important contents of `BaseSegDataset`.

### Initialization of `BaseSegDataset`

Because of `BaseSegDataset` is inherited from [`BaseDataset`](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/basedataset.md)
in MMEngine, it also follows similar parts of initialization process of `BaseDataset`.

Below is a table of some important methods in `BaseSegDataset` initialization method, which would be executed in order from top to bottom.

| Attributes or methods in `BaseSegDataset` initialization |                            Usage                            |
| :------------------------------------------------------: | :---------------------------------------------------------: |
|                     `self._metainfo`                     |              Set meta information of dataset.               |
|                  `self._join_prefix()`                   |            Join paths of images and annotations.            |
|                     `self.pipeline`                      | Build data pipeline for data preprocessing and preparation. |
|                    `self.full_init()`                    |                Full initialize the dataset.                 |

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

Moreover, in initialization of `BaseSegDataset`, `self.label_map` and `self.reduce_zero_label` and would be attributes of `BaseSegDataset`,
whose values `label_map` and `reduce_zero_label` would be added in meta information dict:

The `label_map` is used to require label mapping from old classes in `cls.METAINFO` to new classes in `self._metainfo`,
which changes pixel labels in `load_data_list`.
`label_map` is a dictionary, whose keys are the old label ids and values are the new label ids.
`label_map` is not `None` if and only if (1) old classes in `cls.METAINFO` is not equal to new classes in `self._metainfo`
and (2) both of `cls.METAINFO` and `self._metainfo` are not `None`.

For example, `Cityscapes` dataset usually has 19 classes, users could define new class list by label mapping if they only want to use three classes `road`, `sidewalk` and `building`:

```python
from mmseg.datasets import CityscapesDataset
classes_path = 'new_categories.txt'
# classes.txt with sub categories
categories = ['road', 'sidewalk', 'building']
with open(classes_path, 'w') as f:
    f.write('\n'.join(categories))

train_pipeline = []
kwargs = dict(
    pipeline=train_pipeline,
    data_prefix=dict(img_path='./', seg_map_path='./'),
    metainfo=dict(classes=classes_path))
dataset = CityscapesDataset(**kwargs)
assert list(dataset.metainfo['classes']) == categories
assert dataset.label_map is not None
```

The `reduce_zero_label`(Default to `False`) controls whether to mark label zero as ignored.
Because in semantic segmentation dataset such as [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k),
label `0` usually stands for background, which is not included in classes list in meta information.
If `reduce_zero_label=True`, [`LoadAnnotations`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/datasets/transforms/loading.py#L107-L118)
data transform would ignore label 0 and reduce all other label value by 1.

### Loading meta information

Meta information is collected in `self._metainfo` attribute, which would call [\_load_metainfo](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py#L475) method from MMEngine.
If `metainfo` contains existed filename path, it will be parsed by `list_from_file`, otherwise it is simply parsed as meta information.

| Content of `metainfo` dict                                                      |                            What does `self._metainfo` have?                             |
| :------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------: |
| `metainfo=None`                                                                 |           Load meta information from `cls.METAINFO` defined in dataset class.           |
| `metainfo=dict(classes=('a', 'b'))`                                             | Load class information from `metainfo` dict, palette value would be randomly generated. |
| `metainfo=dict(classes=('a', 'b'), palette=[[100, 100, 100], [200, 200, 200]])` |                Load class and palette information from `metainfo` dict.                 |

Given different `metainfo` dict, `self._metainfo` would call `_load_metainfo`  and returns parsed meta information for train/test process.

### Loading data information

By default, `self.load_data_list()` would be called in `self.full_init()`. In `BaseSegDataset`,
`self.load_data_list()` function is overwritten where annotation path `seg_map_path` would be added from annotation file directory or its meta file.

| Content of `ann_file` dict | Where does `self.load_data_list` load  filename? |
| :------------------------- | :----------------------------------------------: |
| `ann_file=''`              |           Load filename from metafile.           |
| `ann_file=xxx.txt`         |       Load filename from `ann_file` file.        |

If images and annotations folders both have training and validation set, we could define `ann_file` of train and validation set in corresponding dataset class to
overwrite `seg_map_path` in `self.load_data_list` method.

```none
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── xxx{img_suffix}
│   │   │   ├── yyy{img_suffix}
│   │   │   ├── zzz{img_suffix}
│   │   ├── ann_dir
│   │   │   ├── xxx{seg_map_suffix}
│   │   │   ├── yyy{seg_map_suffix}
│   │   │   ├── zzz{seg_map_suffix}
│   │   ├── splits
│   │   │   ├── train.txt
│   │   │   ├── val.txt

```

There are filenames of training and validation dataset in `train.txt` and `val.txt`.

```shell
$ vi train.txt
xxx
yyy

$ vi val.txt
zzz
```

For example, in [`PascalContextDataset` config](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/datasets/pascal_context.py)
`ann_file` of training and validation set are `ImageSets/SegmentationContext/train.txt` and `ImageSets/SegmentationContext/val.txt`, respectively.
Then the `seg_map_path` would be parsed by the content of those two `.txt` file.

### `__getitem__` method

By default, `BaseSegDataset` inherits `__getitem__` method in `BaseDataset`.

In `__getitem__` method, `prepare_data` is called to get the processed data, where data loading pipeline consists of the following steps:

1. Fetch data information by index, implemented by `get_data_info`.
2. Apply data transforms to data, implemented by `pipeline`.

Finally, it returns a dict `data` which includes the idx-th image and data information of dataset after `self.pipeline`.

If you want to implement a new dataset class, you only need to implement `load_data_list` function.
We also encourage users to use the original data loading logic provided by `BaseDataset`.
If the default loading logic is difficult to meet your needs, you can overwrite the `__getitem__` interface to implement your data loading logic.

# Differences between CustomDataset(v0.x)

In MMSegmentation \< 1.x, it uses [`CustomDataset`](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/custom.py#L19) as basic dataset class in MMSegmentation.

`BaseSegDataset` is inherited from `BaseDataset` in MMEngine while `CustomDataset` is inherited from `Dataset` in official PyTorch. The differences between `CustomDataset` and `BaseSegDataset` are lied below:

- `BaseSegDataset` removes all methods about evaluations in `CustomDataset`, such as `format_results`, `pre_eval`, `get_gt_seg_map_by_idx`, `get_gt_seg_maps` and `evaluate`.
- `BaseSegDataset` replaces method `load_annotations` to `load_data_list`.
- `BaseSegDataset` replaces member variable `img_infos` and `split` to `data_list` and `ann_file`, respectively.
- `BaseSegDataset` integrates `CLASSES` and `PALETTE` into two fields of `METAINFO` dict.

# Design your own dataset

All provided datasets class such as `CityscapesDataset` and `ADE20KDataset` are inherited from `BaseSegDataset` in the 1.x version of MMSegmentation.
Some common methods are defined in `BaseSegDataset`, such as loading data information including meta information like categories and palette of dataset,
and data information like path of image and ground truth, and modifying labels of dataset.

If you want to design your own dataset class, you can follow our provided dataset class and add your customization methods.
More information about dataset class may refer to [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/basedataset.md).
