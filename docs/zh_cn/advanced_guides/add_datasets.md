# 新增自定义数据集

## 新增自定义数据集

在这里，我们展示如何构建一个新的数据集。

1. 创建一个新文件 `mmseg/datasets/example.py`

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

2. 在 `mmseg/datasets/__init__.py` 中导入模块

   ```python
   from .example import ExampleDataset
   ```

3. 通过创建一个新的数据集配置文件 `configs/_base_/datasets/example_dataset.py` 来使用它

   ```python
   dataset_type = 'ExampleDataset'
   data_root = 'data/example/'
   ...
   ```

4. 在 `mmseg/utils/class_names.py` 中补充数据集元信息

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

**注意：** 如果新数据集不满足 mmseg 的要求，则需要在 `tools/dataset_converters/` 中准备一个数据集预处理脚本

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

有些数据集不会发布测试集或测试集的标注，如果没有测试集的标注，我们就无法在本地进行评估模型，因此我们在配置文件中将验证集设置为默认测试集。

关于如何构建自己的数据集或实现新的数据集类，请参阅[数据集指南](./datasets.md)以获取更多详细信息。

**注意：** 标注是跟图像同样的形状 (H, W)，其中的像素值的范围是 `[0, num_classes - 1]`。
您也可以使用 [pillow](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#palette) 的 `'P'` 模式去创建包含颜色的标注。

## 通过混合数据去定制数据集

MMSegmentation 同样支持混合数据集去训练。
当前它支持拼接 (concat), 重复 (repeat) 和多图混合 (multi-image mix) 数据集。

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

如果要拼接不同的数据集，可以按如下方式连接数据集配置。

```python
dataset_A_train = dict()
dataset_B_train = dict()
concatenate_dataset = dict(
    type='ConcatDataset',
    datasets=[dataset_A_train, dataset_B_train])
```

下面是一个更复杂的示例，它分别重复 `Dataset_A` 和 `Dataset_B` N 次和 M 次，然后连接重复的数据集。

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

您可以参考 mmengine 的基础数据集[教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/basedataset.html)以了解更多详细信息

### 多图混合集

我们使用 `MultiImageMixDataset` 作为包装（wrapper）去混合多个数据集的图片。
`MultiImageMixDataset`可以被类似 mosaic 和 mixup 的多图混合数据増广使用。

`MultiImageMixDataset` 与 `Mosaic` 数据増广一起使用的例子：

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
