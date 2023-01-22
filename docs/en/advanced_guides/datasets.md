# Dataset

Dataset classes in MMSegmentation have two functions: (1) load data information after [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/user_guides/2_dataset_prepare.md)
and (2) send data into [dataset transform pipeline](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/datasets/basesegdataset.py#L141) to do [data augmentation](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/zh_cn/advanced_guides/transforms.md).
There are 2 kinds of loaded information: (1) meta information which is original dataset information such as categories (classes) of dataset and their corresponding palette information, (2) data information which includes
the path of dataset images and labels.
The tutorial includes some main interfaces in MMSegmentation 1.x dataset class: methods of loading data information and modifying dataset classes in base dataset class, and the relationship between dataset and the data transform pipeline.

## Main Interfaces

Take Cityscapes as an example, if you want to run the example, please download and [preprocess](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/user_guides/2_dataset_prepare.md#cityscapes)
Cityscapes dataset in `data` directory, before running the demo code:

Instantiate Cityscapes training dataset:

```python
from mmseg.datasets import CityscapesDataset
from mmseg.utils import register_all_modules
register_all_modules()

data_root = 'data/cityscapes/'
data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

dataset = CityscapesDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline)
```

Get the length of training set:

```python
print(len(dataset))

2975
```

Get data information: The type of data information is `dict` which includes several keys:

- `'img_path'`: path of images
- `'seg_map_path'`: path of segmentation labels
- `'seg_fields'`: saving label fields
- `'sample_idx'`: the  index of the current sample

There are also `'label_map'` and `'reduce_zero_label'` whose functions would be introduced in the next section.

```python
# Acquire data information of first sample in dataset
print(dataset.get_data_info(0))

{'img_path': 'data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',
 'seg_map_path': 'data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png',
 'label_map': None,
 'reduce_zero_label': False,
 'seg_fields': [],
 'sample_idx': 0}
```

Get dataset meta information: the type of MMSegmentation meta information is also `dict`, which includes `'classes'` field for dataset classes and `'palette'` field for corresponding colors in visualization, and has `'label_map'` field and `'reduce_zero_label'` filed.

```python
print(dataset.metainfo)

{'classes': ('road',
  'sidewalk',
  'building',
  'wall',
  'fence',
  'pole',
  'traffic light',
  'traffic sign',
  'vegetation',
  'terrain',
  'sky',
  'person',
  'rider',
  'car',
  'truck',
  'bus',
  'train',
  'motorcycle',
  'bicycle'),
 'palette': [[128, 64, 128],
  [244, 35, 232],
  [70, 70, 70],
  [102, 102, 156],
  [190, 153, 153],
  [153, 153, 153],
  [250, 170, 30],
  [220, 220, 0],
  [107, 142, 35],
  [152, 251, 152],
  [70, 130, 180],
  [220, 20, 60],
  [255, 0, 0],
  [0, 0, 142],
  [0, 0, 70],
  [0, 60, 100],
  [0, 80, 100],
  [0, 0, 230],
  [119, 11, 32]],
 'label_map': None,
 'reduce_zero_label': False}
```

The return value of dataset `__getitem__` method is the output of data samples after data augmentation, whose type is also `dict`. It has two fields: `'inputs'` corresponding to images after data augmentation,
and `'data_samples'` corresponding to `SegDataSample`\](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/zh_cn/advanced_guides/structures.md) which is new data structures in MMSegmentation 1.x,
and `gt_sem_seg` of `SegDataSample` has labels after data augmentation operations.

```python
print(dataset[0])

{'inputs': tensor([[[131, 130, 130,  ...,  23,  23,  23],
          [132, 132, 132,  ...,  23,  22,  23],
          [134, 133, 133,  ...,  23,  23,  23],
          ...,
          [ 66,  67,  67,  ...,  71,  71,  71],
          [ 66,  67,  66,  ...,  68,  68,  68],
          [ 67,  67,  66,  ...,  70,  70,  70]],

         [[143, 143, 142,  ...,  28,  28,  29],
          [145, 145, 145,  ...,  28,  28,  29],
          [145, 145, 145,  ...,  27,  28,  29],
          ...,
          [ 75,  75,  76,  ...,  80,  81,  81],
          [ 75,  76,  75,  ...,  80,  80,  80],
          [ 77,  76,  76,  ...,  82,  82,  82]],

         [[126, 125, 126,  ...,  21,  21,  22],
          [127, 127, 128,  ...,  21,  21,  22],
          [127, 127, 126,  ...,  21,  21,  22],
          ...,
          [ 63,  63,  64,  ...,  69,  69,  70],
          [ 64,  65,  64,  ...,  69,  69,  69],
          [ 65,  66,  66,  ...,  72,  71,  71]]], dtype=torch.uint8),
 'data_samples': <SegDataSample(

     META INFORMATION
     img_path: 'data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
     seg_map_path: 'data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'
     img_shape: (512, 1024, 3)
     flip_direction: None
     ori_shape: (1024, 2048)
     flip: False

     DATA FIELDS
     gt_sem_seg: <PixelData(

             META INFORMATION

             DATA FIELDS
             data: tensor([[[2, 2, 2,  ..., 8, 8, 8],
                          [2, 2, 2,  ..., 8, 8, 8],
                          [2, 2, 2,  ..., 8, 8, 8],
                          ...,
                          [0, 0, 0,  ..., 0, 0, 0],
                          [0, 0, 0,  ..., 0, 0, 0],
                          [0, 0, 0,  ..., 0, 0, 0]]])
         )>
     _gt_sem_seg: <PixelData(

             META INFORMATION

             DATA FIELDS
             data: tensor([[[2, 2, 2,  ..., 8, 8, 8],
                          [2, 2, 2,  ..., 8, 8, 8],
                          [2, 2, 2,  ..., 8, 8, 8],
                          ...,
                          [0, 0, 0,  ..., 0, 0, 0],
                          [0, 0, 0,  ..., 0, 0, 0],
                          [0, 0, 0,  ..., 0, 0, 0]]])
         )>
 )}
```

## BaseSegDataset

As mentioned above, dataset classes have the same functions, we implemented  [`BaseSegDataset`](https://mmsegmentation.readthedocs.io/en/dev-1.x/api.html?highlight=BaseSegDataset#mmseg.datasets.BaseSegDataset) to reues the common functions.
It inherits [`BaseDataset` of MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/basedataset.md) and follows unified initialization process of OpenMMLab. It supports the highly effective interior storing format, some functions like
dataset concatenation and repeatedly sampling. In MMSegmentation `BaseSegDataset`, the **method of loading data information** (`load_data_list`) is redefined and adds new `get_label_map` method to **modify dataset classes information**.

### Loading Dataset Information

The loaded data information includes the path of images samples and annotations samples, the detailed implementation could be found in
[`load_data_list`](https://github.com/open-mmlab/mmsegmentation/blob/163277bfe0fa8fefb63ee5137917fafada1b301c/mmseg/datasets/basesegdataset.py#L231) of `BaseSegDataset` in MMSegmentation.
There are two main methods to acquire the path of images and labels:

1. Load file paths according to the dirictory and suffix of input images and annotations

If the dataset directory structure is organized as below, the [`load_data_list`](https://github.com/open-mmlab/mmsegmentation/blob/163277bfe0fa8fefb63ee5137917fafada1b301c/mmseg/datasets/basesegdataset.py#L231) can parse dataset directory Structure:

```
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

Here is an example pf ADE20K, and below the directory structure of the dataset:

```
├── ade
│   ├── ADEChallengeData2016
│   │   ├── annotations
│   │   │   ├── training
│   │   │   │   ├── ADE_train_00000001.png
│   │   │   │   ├── ...
│   │   │   │── validation
│   │   │   │   ├── ADE_val_00000001.png
│   │   │   │   ├── ...
│   │   ├── images
│   │   │   ├── training
│   │   │   │   ├── ADE_train_00000001.jpg
│   │   │   │   ├── ...
│   │   │   ├── validation
│   │   │   │   ├── ADE_val_00000001.jpg
│   │   │   │   ├── ...
```

```python
from mmseg.datasets import ADE20KDataset

ADE20KDataset(data_root = 'data/ade/ADEChallengeData2016',
    data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
    img_suffix='.jpg',
    seg_map_suffix='.png',
    reduce_zero_label=True)
```

2. Load file paths from annotation file

Dataset also can load an annotation file which includes the data sample paths of dataset.
Take PascalContext dataset instance as an example, its input annotation file is:

```python
2008_000008
...
```

It needs to define `ann_file` when instantiation:

```python
PascalContextDataset(data_root='data/VOCdevkit/VOC2010/',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
    ann_file='ImageSets/SegmentationContext/train.txt')
```

### Modification of Dataset Classes

- Use `metainfo` input argument

Meta information is defined as class variables, such as `METAINFO` variable of Cityscapes:

```python
class CityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

```

Here `'classes'` defines class names of Cityscapes dataset annotations, if users only concern some classes about vehicles and **ignore other classes**,
the meta information of dataset could be modified by defined input argument `metainfo` when instantiating Cityscapes dataset:

```python
from mmseg.datasets import CityscapesDataset

data_root = 'data/cityscapes/'
data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train')
# metainfo only keep classes below:
metainfo=dict(classes=( 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'))
dataset = CityscapesDataset(data_root=data_root, data_prefix=data_prefix, metainfo=metainfo)

print(dataset.metainfo)

{'classes': ('car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'),
 'palette': [[0, 0, 142],
  [0, 0, 70],
  [0, 60, 100],
  [0, 80, 100],
  [0, 0, 230],
  [119, 11, 32],
  [128, 64, 128],
  [244, 35, 232],
  [70, 70, 70],
  [102, 102, 156],
  [190, 153, 153],
  [153, 153, 153],
  [250, 170, 30],
  [220, 220, 0],
  [107, 142, 35],
  [152, 251, 152],
  [70, 130, 180],
  [220, 20, 60],
  [255, 0, 0]],
 # pixels whose label index are 255 would be ignored when calculating loss
 'label_map': {0: 255,
  1: 255,
  2: 255,
  3: 255,
  4: 255,
  5: 255,
  6: 255,
  7: 255,
  8: 255,
  9: 255,
  10: 255,
  11: 255,
  12: 255,
  13: 0,
  14: 1,
  15: 2,
  16: 3,
  17: 4,
  18: 5},
 'reduce_zero_label': False}
```

Meta information is different from default setting of Cityscapes dataset. Moreover, `label_map` field is also defined, which is used for modifying label index of each pixel on segmentation mask.
The segmentation label would re-map class information by `label_map`, [here](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/datasets/basesegdataset.py#L151) is detailed implementation:

```python
gt_semantic_seg_copy = gt_semantic_seg.copy()
for old_id, new_id in results['label_map'].items():
    gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
```

- Using `reduce_zero_label` input argument

To ignore label 0 (such as ADE20K dataset), we can use `reduce_zero_label` (default to `False`) argument of BaseSegDataset and its subclasses.
When `reduce_zero_label` is `True`, label 0 in segmentation annotations would be set as 255 (models of MMSegmentation would ignore label 255 in calculating loss) and indices of other labels will minus 1:

```python
gt_semantic_seg[gt_semantic_seg == 0] = 255
gt_semantic_seg = gt_semantic_seg - 1
gt_semantic_seg[gt_semantic_seg == 254] = 255
```

## Dataset and Data Transform Pipeline

If the argument `pipeline` is defined, the return value of `__getitem__` method is after data argument.
If dataset input argument does not define pipeline, it is the same as return value of `get_data_info` method.

```python
from mmseg.datasets import CityscapesDataset

data_root = 'data/cityscapes/'
data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train')
dataset = CityscapesDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False)

print(dataset[0])

{'img_path': 'data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',
 'seg_map_path': 'data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png',
 'label_map': None,
 'reduce_zero_label': False,
 'seg_fields': [],
 'sample_idx': 0}
```
