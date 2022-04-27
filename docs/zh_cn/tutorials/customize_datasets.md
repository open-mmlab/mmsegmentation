# 教程 2: 自定义数据集

## 如何准备自己的数据集

准备自己的数据集可以分成一下几个步骤：

- 数据集准备，将数据集生成规定的格式，并存放到正确的位置，如 `./data/`。
- 在 `./mmseg/datasets/` 里定义该数据集以注册到 `DATASETS` 里。
- 在 `./configs/_base_/datasets/` 里面设置训练与验证时的各种超参数，如数据增强策略，图像裁剪大小，图像正则化等等。

主要改动的文件位置为：

```none
mmsegmentation
   |
   |- data
   |     |- my_dataset                 # 转换后的自己的数据集文件
   |- mmseg
   |     |- datasets
   |     |     |- __init__.py          # 在这里加入自己的数据集的类
   |     |     |- my_dataset.py               ## 定义自己的数据集的类
   |     |     |- ...
   |- configs
   |     |- _base_
   |     |     |- datasets
   |     |     |     |- my_dataset_config.py      # 自己的数据集的配置文件
   |     |     |- ...
   |     |- ...
   |- ...
```

### 数据集的准备

首先将自己的数据集处理成下面的格式，其中`img_suffix` 和 `seg_map_suffix` 是图像和注释的格式，常用的是 `.png` 和 `.jpg`。

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

### 注册数据集

生成好上述数据格式后，在 `./mmseg/dataset/` 里新建文件 `my_dataset.py`，使它里面定义的 `MyDataset` 类可以被注册到 MMCV 的 `DATASETS` 里面，然后被模型调用：


```python
from .builder import DATASETS
from .custom import CustomDataset

#将 MyDataset 类注册到 DATASETS 里
@DATASETS.register_module()
class MyDataset(CustomDataset):
    # 数据集标注的各类名称，即 0, 1, 2, 3... 各个类别的对应名称
    CLASSES = ('background', 'label_a', 'label_b', 'label_c',
               'label_d', ...)
    # 各类类别的 BGR 三通道值，用于可视化预测结果
    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], ...]

    # 图片和对应的标注，这里对应的文件夹下均为 .png 格式
    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False, # 此时 label 里的 0 是背景（上面 CLASSES 里第一个），所以这里是 False
            **kwargs)
```

在`./mmseg/dataset/my_dataset.py`里面定义了数据集的分割类别 `CLASSES` 和在对应的 BGR 通道的调色板 `PALETTE`，`PALETTE` 只在预测结果可视化的时候会用到，并不会影响训练和验证。

需要强调的是，如果 label 中的 0 并不是背景，那么需要设置 `reduce_zero_label=True`。

创建好`./mmseg/dataset/my_dataset.py`后，需要在 `./mmseg/dataset/__init__.py`里也加入它：

```python
# Copyright (c) OpenMMLab. All rights reserved.
from .my_dataset import MyDataset

__all__ = [
    ...,
    'MyDataset'
]
```

### 设置数据集配置文件

数据集定义好后，还需要在 `./configs/_base_/datasets/` 里面定义该数据集有关的配置项 `my_dataset_config.py`，使之与其他的配置参数一起在训练和测试时调用。

```python
# 在./mmseg/datasets/__init__.py 中定义的数据集类型
dataset_type = 'MyDataset'
# 数据集准备生成的文件夹路径
data_root = 'data/my_dataset'

img_norm_cfg = dict( # 常用这组参数归一化是因为它是 ImageNet 1K 预训练使用的图像均值与方差
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (512, 512) # 训练时图像裁剪的大小
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
    samples_per_gpu=4, # 单个 GPU 的 Batch size
    workers_per_gpu=4, # 单个 GPU 分配的数据加载线程数
    train=dict( # 训练数据集配置
        type=dataset_type, # 数据集的类别, 细节参考自 mmseg/datasets/
        data_root=data_root, # 数据集的根目录。
        img_dir='images/training', # 数据集图像的文件夹
        ann_dir='annotations/training', # 数据集注释的文件夹
        pipeline=train_pipeline), # 流程， 由之前创建的 train_pipeline 传递进来
    val=dict( # 验证数据集的配置
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline), # 由之前创建的 test_pipeline 传递的流程
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))

```
各个配置项的具体作用可以参考[配置文件教程](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/tutorials/config.md) 。

我们做好了数据集的准备，只需要让模型的配置文件里导入 `./configs/_base_/datasets/my_dataset_config.py`即可使用这个数据集，例如：

```python
_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/my_dataset_config.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=YOUR_DATASET_CLASSES), auxiliary_head=dict(num_classes=YOUR_DATASET_CLASSES))

```

至此，您可以在 MMSegmentation 里使用自己的数据集了。

## 通过重新组织数据来定制数据集

最简单的方法是将您的数据集进行转化，并组织成文件夹的形式。

如下的文件结构就是一个例子。

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

一个训练对将由 img_dir/ann_dir 里同样首缀的文件组成。

如果给定 `split` 参数，只有部分在 img_dir/ann_dir 里的文件会被加载。
我们可以对被包括在 split 文本里的文件指定前缀。

除此以外，一个 split 文本如下所示：

```none
xxx
zzz
```

只有

`data/my_dataset/img_dir/train/xxx{img_suffix}`,
`data/my_dataset/img_dir/train/zzz{img_suffix}`,
`data/my_dataset/ann_dir/train/xxx{seg_map_suffix}`,
`data/my_dataset/ann_dir/train/zzz{seg_map_suffix}` 将被加载。

注意：标注是跟图像同样的形状 (H, W)，其中的像素值的范围是 `[0, num_classes - 1]`。
您也可以使用 [pillow](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#palette) 的 `'P'` 模式去创建包含颜色的标注。

## 通过混合数据去定制数据集

MMSegmentation 同样支持混合数据集去训练。
当前它支持拼接 (concat), 重复 (repeat) 和多图混合 (multi-image mix)数据集。

### 重复数据集

我们使用 `RepeatDataset` 作为包装 (wrapper) 去重复数据集。
例如，假设原始数据集是 `Dataset_A`，为了重复它，配置文件如下：

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # 这是 Dataset_A 数据集的原始配置
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### 拼接数据集

有2种方式去拼接数据集。

1. 如果您想拼接的数据集是同样的类型，但有不同的标注文件，
    您可以按如下操作去拼接数据集的配置文件：

    1. 您也许可以拼接两个标注文件夹 `ann_dir`

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = ['anno_dir_1', 'anno_dir_2'],
            pipeline=train_pipeline
        )
        ```

    2. 您也可以去拼接两个 `split` 文件列表

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = 'anno_dir',
            split = ['split_1.txt', 'split_2.txt'],
            pipeline=train_pipeline
        )
        ```

    3. 您也可以同时拼接 `ann_dir` 文件夹和 `split` 文件列表

        ```python
        dataset_A_train = dict(
            type='Dataset_A',
            img_dir = 'img_dir',
            ann_dir = ['anno_dir_1', 'anno_dir_2'],
            split = ['split_1.txt', 'split_2.txt'],
            pipeline=train_pipeline
        )
        ```

        在这样的情况下， `ann_dir_1` 和 `ann_dir_2` 分别对应于 `split_1.txt` 和 `split_2.txt`

2. 如果您想拼接不同的数据集，您可以如下去拼接数据集的配置文件：

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

一个更复杂的例子如下：分别重复 `Dataset_A` 和 `Dataset_B` N 次和 M 次，然后再去拼接重复后的数据集

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

### 多图混合集

我们使用 `MultiImageMixDataset` 作为包装(wrapper)去混合多个数据集的图片。
`MultiImageMixDataset`可以被类似mosaic和mixup的多图混合数据増广使用。

`MultiImageMixDataset`与`Mosaic`数据増广一起使用的例子：

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
