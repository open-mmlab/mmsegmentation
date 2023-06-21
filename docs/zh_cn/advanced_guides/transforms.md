# 数据增强变化

在本教程中，我们将介绍 MMSegmentation 中数据增强变化流程的设计。

本指南的结构如下：

- [数据增强变化](#数据增强变化)
  - [数据增强变化流程设计](#数据增强变化流程设计)
    - [数据加载](#数据加载)
    - [预处理](#预处理)
    - [格式修改](#格式修改)

## 数据增强变化流程设计

按照惯例，我们使用 `Dataset` 和 `DataLoader` 多进程地加载数据。`Dataset` 返回与模型 forward 方法的参数相对应的数据项的字典。由于语义分割中的数据可能大小不同，我们在 MMCV 中引入了一种新的 `DataContainer` 类型，以帮助收集和分发不同大小的数据。参见[此处](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py)了解更多详情。

在 MMSegmentation 的 1.x 版本中，所有数据转换都继承自 [`BaseTransform`](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/transforms/base.py#L6).

转换的输入和输出类型都是字典。一个简单的示例如下：

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

数据准备流程和数据集是解耦的。通常，数据集定义如何处理标注，数据流程定义准备数据字典的所有步骤。流程由一系列操作组成。每个操作都将字典作为输入，并为接下来的转换输出字典。

操作分为数据加载、预处理、格式修改和测试数据增强。

这里是 PSPNet 的流程示例：

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

对于每个操作，我们列出了 `添加`/`更新`/`删除` 相关的字典字段。在流程前，我们可以从数据集直接获得的信息是 `img_path` 和 `seg_map_path`。

### 数据加载

`LoadImageFromFile`：从文件加载图像。

- 添加：`img`，`img_shape`，`ori_shape`

`LoadAnnotations`：加载数据集提供的语义分割图。

- 添加：`seg_fields`，`gt_seg_map`

### 预处理

`RandomResize`：随机调整图像和分割图大小。

-添加：`scale`，`scale_factor`，`keep_ratio`
-更新：`img`，`img_shape`，`gt_seg_map`

`Resize`：调整图像和分割图的大小。

-添加：`scale`，`scale_factor`，`keep_ratio`
-更新：`img`，`gt_seg_map`，`img_shape`

`RandomCrop`：随机裁剪图像和分割图。

-更新：`img`，`gt_seg_map`，`img_shape`

`RandomFlip`：翻转图像和分割图。

-添加：`flip`，`flip_direction`
-更新：`img`，`gt_seg_map`

`PhotoMetricDistortion`：按顺序对图像应用光度失真，每个变换的应用概率为 0.5。随机对比度的位置是第二或倒数第二（分别为下面的模式 0 或 1）。

```
1.随机亮度
2.随机对比度（模式 0）
3.将颜色从 BGR 转换为 HSV
4.随机饱和度
5.随机色调
6.将颜色从 HSV 转换为 BGR
7.随机对比度（模式 1）
```

- 更新：`img`

### 格式修改

`PackSegInputs`：为语义分段打包输入数据。

- 添加：`inputs`，`data_sample`
- 删除：由 `meta_keys` 指定的 keys（合并到 data_sample 的 metainfo 中），所有其他 keys
