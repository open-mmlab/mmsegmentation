# 数据集

在 MMSegmentation 算法库中, 所有 Dataset 类的功能有两个: 加载[预处理](../user_guides/2_dataset_prepare.md) 之后的数据集的信息, 和将数据送入[数据集变换流水线](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/basesegdataset.py#L141) 中, 进行[数据变换操作](./transforms.md). 加载的数据集信息包括两类: 元信息 (meta information), 数据集本身的信息, 例如数据集总共的类别, 和它们对应调色盘信息: 数据信息 (data information) 是指每组数据中图片和对应标签的路径. 下文中介绍了 MMSegmentation 1.x 中数据集的常用接口, 和 mmseg 数据集基类中数据信息加载与修改数据集类别的逻辑, 以及数据集与数据变换流水线 (pipeline) 的关系.

## 常用接口

以 Cityscapes 为例, 介绍数据集常用接口. 如需运行以下示例, 请在当前工作目录下的 `data` 目录下载并[预处理](../user_guides/2_dataset_prepare.md#cityscapes) Cityscapes 数据集.

实例化 Cityscapes 训练数据集:

```python
from mmengine.registry import init_default_scope
from mmseg.datasets import CityscapesDataset

init_default_scope('mmseg')

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

查看训练数据集长度:

```python
print(len(dataset))

2975
```

获取数据信息, 数据信息的类型是一个字典, 包括 `'img_path'` 字段的存放图片的路径和 `'seg_map_path'` 字段存放分割标注的路径, 以及标签重映射的字段 `'label_map'` 和 `'reduce_zero_label'`(主要功能在下文中介绍), 还有存放已加载标签字段 `'seg_fields'`, 和当前样本的索引字段 `'sample_idx'`.

```python
# 获取数据集中第一组样本的数据信息
print(dataset.get_data_info(0))

{'img_path': 'data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',
 'seg_map_path': 'data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png',
 'label_map': None,
 'reduce_zero_label': False,
 'seg_fields': [],
 'sample_idx': 0}
```

获取数据集元信息, MMSegmentation 的数据集元信息的类型同样是一个字典, 包括 `'classes'` 字段存放数据集类别, `'palette'` 存放数据集类别对应的可视化时调色盘的颜色, 以及标签重映射的字段 `'label_map'` 和 `'reduce_zero_label'`.

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

数据集 `__getitem__` 方法的返回值, 是经过数据增强的样本数据的输出, 同样也是一个字典, 包括两个字段, `'inputs'` 字段是当前样本经过数据增强操作的图像, 类型为 torch.Tensor, `'data_samples'` 字段存放的数据类型是 MMSegmentation 1.x 新添加的数据结构 [`Segdatasample`](./structures.md), 其中`gt_sem_seg` 字段是经过数据增强的标签数据.

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

由于 MMSegmentation 中的所有数据集的基本功能均包括(1) 加载[数据集预处理](../user_guides/2_dataset_prepare.md) 之后的数据信息和 (2) 将数据送入数据变换流水线中进行数据变换, 因此在 MMSegmentation 中将其中的共同接口抽象成 [`BaseSegDataset`](https://mmsegmentation.readthedocs.io/zh_CN/latest/api.html?highlight=BaseSegDataset#mmseg.datasets.BaseSegDataset)，它继承自 [MMEngine 的 `BaseDataset`](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/basedataset.md), 遵循 OpenMMLab 数据集初始化统一流程, 支持高效的内部数据存储格式, 支持数据集拼接、数据集重复采样等功能.
在 MMSegmentation BaseSegDataset 中重新定义了**数据信息加载方法**（`load_data_list`）和并新增了 `get_label_map` 方法用来**修改数据集的类别信息**.

### 数据信息加载

数据信息加载的内容是样本数据的图片路径和标签路径, 具体实现在 MMSegmentation 的 BaseSegDataset 的 [`load_data_list`](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/basesegdataset.py#L231) 中.
主要有两种获取图片和标签的路径方法, 如果当数据集目录按以下目录结构组织, [`load_data_list`](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/basesegdataset.py#L231)) 会根据数据路径和后缀来解析.

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

例如 ADE20k 数据集结构如下所示:

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

实例化 ADE20k 数据集时，输入图片和标签的路径和后缀:

```python
from mmseg.datasets import ADE20KDataset

ADE20KDataset(data_root = 'data/ade/ADEChallengeData2016',
    data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
    img_suffix='.jpg',
    seg_map_suffix='.png',
    reduce_zero_label=True）
```

如果数据集有标注文件, 实例化数据集时会根据输入的数据集标注文件加载数据信息. 例如, PascalContext 数据集实例, 输入标注文件的内容为:

```python
2008_000008
...
```

实例化时需要定义 `ann_file`

```python
PascalContextDataset(data_root='data/VOCdevkit/VOC2010/',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
    ann_file='ImageSets/SegmentationContext/train.txt')
```

### 数据集类别修改

- 通过输入 metainfo 修改
  `BaseSegDataset` 的子类元信息在数据集实现时定义为类变量，例如 Cityscapes 的 `METAINFO` 变量:

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

这里的 `'classes'` 中定义了 Cityscapes 数据集标签中的类别名, 如果训练时只关注几个交通工具类别, **忽略其他类别**,
在实例化 Cityscapes 数据集时通过定义 `metainfo` 输入参数的 classes 的字段来修改数据集的元信息:

```python
from mmseg.datasets import CityscapesDataset

data_root = 'data/cityscapes/'
data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train')
# metainfo 中只保留以下 classes
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
 # 类别索引为 255 的像素，在计算损失时会被忽略
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

可以看到, 数据集元信息的类别和默认 Cityscapes 不同. 并且, 定义了标签重映射的字段 `label_map` 用来修改每个分割掩膜上的像素的类别索引, 分割标签类别会根据 `label_map`, 将类别重映射, [具体实现](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/basesegdataset.py#L151):

```python
gt_semantic_seg_copy = gt_semantic_seg.copy()
for old_id, new_id in results['label_map'].items():
    gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
```

- 通过 `reduce_zero_label` 修改
  对于常见的忽略 0 号标签的场景, `BaseSegDataset` 的子类中可以用 `reduce_zero_label` 输入参数来控制。`reduce_zero_label` (默认为 `False`)
  用来控制是否将标签 0 忽略, 当该参数为 `True` 时(最常见的应用是 ADE20k 数据集), 对分割标签中第 0 个类别对应的类别索引改为 255 (MMSegmentation 模型中计算损失时, 默认忽略 255), 其他类别对应的类别索引减一:

```python
gt_semantic_seg[gt_semantic_seg == 0] = 255
gt_semantic_seg = gt_semantic_seg - 1
gt_semantic_seg[gt_semantic_seg == 254] = 255
```

## 数据集与数据变换流水线

在常用接口的例子中可以看到, 输入的参数中定义了数据变换流水线参数 `pipeline`, 数据集 `__getitem__` 方法返回经过数据变换的值.
当数据集输入参数没有定义 pipeline, 返回值和 `get_data_info` 方法返回值相同, 例如:

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
