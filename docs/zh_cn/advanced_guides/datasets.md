# 数据集

在 OpenMMLab 2.0 里面, MMEngine 实现了 `BaseDataset` 类来给 OpenMMLab 下游各个算法库提供一些基础的数据集相关接口, 下游的算法库可以继承这个基础的数据集类并增加自定义的方法.
MMEngine `BaseDataset` 的基础功能是加载数据集的信息, 它主要包括两种. 一种是元信息 (meta information), 包括了数据集本身的信息, 它有时候会被模型和外部组件获取到. 另一种是数据信息 (data information), 例如图像和注释文件的路径.
更多关于 `BaseDataset` 的细节可以在 [MMEngine BaseDataset 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/basedataset.md) 里面找到.

`BaseSegDataset` 是 MMSegmentation v1.x 里面用于语义分割任务的数据集基础类, 它继承自 MMEngine 的 `BaseDataset`.

在这个文档里, 我们将介绍 MMSegmentation 里面的 `BaseSegDataset` 类. 首先, 我们将介绍 MMSegmentation 里面标准的数据集格式. 然后, 我们将介绍
`BaseSegDataset` 类里面的一些重要方法/函数:

- (1) `BaseSegDataset` 的初始化, 以及在初始化过程中被调用的具体方法.

- (2) 加载元信息和原始数据信息, 在训练和测试过程中, 这两种信息的加载函数是 `BaseSegDataset` 的基础函数.

- (3) `__getitem__` 方法, 和 `torch.utils.data.Dataset` 里面的 `__getitem__` 类似, 它被用来在 `BaseSegDataset` 初始化时给数据信息索引.

关于 `BaseSegDataset` 的更多信息可以在 [basesegdataset.py](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/datasets/basesegdataset.py) 里找到.

## 数据集格式

`BaseSegDataset` 被用来处理全监督语义分割任务, 因此在训练时图像和标注都是需要的. 对应的存放数据集的文件夹结构如下所示:

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

`BaseSegDataset` 规定数据集图像和对应的标注应该有同样的文件名, 除了后缀 (suffix), 在 MMSegmentation 里定制化数据集的类也需要遵照这样的规则.

以 [`CityscapesDataset`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/datasets/cityscapes.py#L7) 数据集类为例:

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

从上面我们可以发现一个合法的图像和对应的标注注释文件名应该是 `xxx{img_suffix}`
和 `xxx{seg_map_suffix}` (文件属性如 `.png` 同样被包括在后缀里), 在 `CityscapesDataset` 里面分别是 `img_suffix='_leftImg8bit.png'` 和 `seg_map_suffix='_gtFine_labelTrainIds.png'`.
`get_data_info` 和 `metainfo` 同样被 `BaseSegDataset` 引入用来保存原始数据信息 (例如图像的路径) 和元数据信息 (例如注释里分割的类别名称).

接下来, 我们将介绍 `BaseSegDataset` 里面的一些重要内容.

### `BaseSegDataset` 的初始化

因为 `BaseSegDataset` 继承自 MMEngine 里面的 [`BaseDataset`](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/basedataset.md),
它同样遵循 `BaseDataset` 里初始化的部分过程.

下表列出了 `BaseSegDataset` 初始化里的一些重要属性和方法, 在初始化时将会从上到下按顺序执行.

| `BaseSegDataset` 初始化里的属性或方法 |                           作用                            |
| :-----------------------------------: | :-------------------------------------------------------: |
|           `self._metainfo`            |                     设置数据集元信息.                     |
|         `self._join_prefix()`         |               连接 (join) 图像和注释的路径.               |
|            `self.pipeline`            | 构建用于数据预处理和数据准备的数据流水线 (data pipeline). |
|          `self.full_init()`           |                     完全初始化数据集.                     |

在 `BaseSegDataset` 里面, 参数 `lazy_init` (默认设置为 False) 被用来控制是否在实例化期间加载注释.
如果不使用懒加载 (lazy initialization), `self.full_init()` 将被调用, 它会从上到下的执行下面的一些方法:

|  `self.full_init()` 里执行的方法  |                作用                 |
| :-------------------------------: | :---------------------------------: |
|      `self.load_data_list()`      |            加载数据信息.            |
|       `self.filter_data()`        |  过滤非法数据, 例如没有标注的数据.  |
| `self._get_unserialized_subset()` |      根据索引来得到数据的子集.      |
|     `self._serialize_data()`      | 序列化对应数据集类里的 `data_list`. |

在某些情况, 如预测的可视化时, 只需要用数据的元信息, 而加载注释文件是不必要的.
通过设置 `lazy_init=True`, `BaseSegDataset` 可以跳过加载和解析注释以节约时间. 在这种情况下 `self.full_init()` 将不会被执行.

除此之外, 在 `BaseSegDataset` 初始化时, `self.label_map` 和 `self.reduce_zero_label` 将成为 `BaseSegDataset` 的属性,
它们的值 `label_map` 和 `reduce_zero_label` 将别添加到元信息字典里.

`label_map` 被用来获取标签映射 (label mapping), 从 `cls.METAINFO` 里的旧类别到 `self._metainfo` 里的新类别. 它会改变 `load_data_list`
里的像素类别. `label_map` 是一个字典, 它的键是旧的标签的 id 而值是新的标签的 id.
`label_map` 不是 `None` 当且仅当: (1) `cls.METAINFO` 的旧的类别不等于 `self._metainfo` 里的新的类别, 并且 (2) `cls.METAINFO` 和 `self._metainfo` 都不是 `None` 同时成立.

例如, `Cityscapes` 数据集通常含有 19 个类别, 使用者能够通过标签映射定义新的类别 如果他们只想用 `road`, `sidewalk` 和 `building` 这三个类别:

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

`reduce_zero_label`(默认设置为 `False`) 用来控制是否将标签 0 标为被忽略的.
因为在语义分割数据集例如 [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k) 里面,
标签 `0` 通常表示背景, 它没有被包含在元信息的类别里面.
如果 `reduce_zero_label=True`, [`LoadAnnotations`](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/mmseg/datasets/transforms/loading.py#L107-L118)
数据变换将忽略标签 0 并且将其他所有类别的值减去 1, 除了默认的被忽略的类别索引 (默认为 `255`).

### 加载元信息

元信息在属性 `self._metainfo` 里被收集, 它将调用 MMEngine 里面的 [\_load_metainfo](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py#L475) 方法.
如果 `metainfo` 包含了存在的文件路径, 它将被 `list_from_file` 解析, 否则它将被简单解析成元信息.

| `metainfo` 字典里的内容                                                         |                         `self._metainfo` 里有什么?                         |
| :------------------------------------------------------------------------------ | :------------------------------------------------------------------------: |
| `metainfo=None`                                                                 |              从定义在数据集类里的 `cls.METAINFO` 加载元信息.               |
| `metainfo=dict(classes=('a', 'b'))`                                             | 从 `metainfo` 字典里加载类别 (class)信息, 调色盘 (palette) 的值将随机生成. |
| `metainfo=dict(classes=('a', 'b'), palette=[[100, 100, 100], [200, 200, 200]])` |     从 `metainfo` 字典里加载类别信息 (class) 和调色盘 (palette) 的值.      |

给定 `metainfo` 字典, `self._metainfo` 将调用 `_load_metainfo`  并返回解析好的元信息, 用于训练和测试过程.

### 加载数据信息

默认情况下, `self.load_data_list()` 将在 `self.full_init()` 里面被调用. 在 `BaseSegDataset` 里面,
`self.load_data_list()` 函数被重写, 其中注释的路径 `seg_map_path` 会从注释文件的所在文件夹里或数据集本身的元文件里得到, 并被添加到返回值 `data_list`.

| `ann_file` 字典里的内容 | `self.load_data_list` 从哪里加载文件名? |
| :---------------------- | :-------------------------------------: |
| `ann_file=''`           |          从元文件里加载文件名.          |
| `ann_file=xxx.txt`      |     从 `ann_file` 文件里加载文件名.     |

如果图像和注释的文件夹里 **同时有训练和验证集**, 我们可以在对应的数据集类里定义训练集和验证集的 `ann_file` 来重写 `self.load_data_list` 方法里的 `seg_map_path`.

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

`train.txt` 和 `val.txt` 里面有训练集和验证集的文件名.

```shell
$ vi train.txt
xxx
yyy

$ vi val.txt
zzz
```

例如, 在 [`PascalContextDataset` 配置文件](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/configs/_base_/datasets/pascal_context.py) 里面,
训练集和验证集的 `ann_file` 分别是 `ImageSets/SegmentationContext/train.txt` 和 `ImageSets/SegmentationContext/val.txt`.
随后 `seg_map_path` 将被解析为这两个 `.txt` 里面的内容.

### `__getitem__` 方法

默认情况下, `BaseSegDataset` 继承了 `BaseDataset` 里的 `__getitem__` 方法.

在 `__getitem__` 方法里, `prepare_data` 被用来获得处理好的数据, 其中的数据加载流水线 (data loading pipeline) 包含以下几步:

1. 通过索引 (index) 获取数据信息, 由 `get_data_info` 实现.
2. 对数据进行数据变换 (data transforms), 由 `pipeline` 实现.

最终, 它返回一个 `data` 字典, 包含了第 idx 个图像和经过 `self.pipeline` 进行数据变换后的数据信息.

如果你想要实现一个新的数据集的类, 你可能只需要定义一个新的 `load_data_list` 方法.
我们推荐使用者遵循 `BaseDataset` 提供的原始数据加载逻辑. 如果默认的加载逻辑很那满足你的实际需求, 你可以重写 `__getitem__` 方法来实现你自己的数据加载逻辑.

# 和 CustomDataset(v0.x) 的区别

在 MMSegmentation \< 1.x 里面, 它使用 [`CustomDataset`](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/custom.py#L19) 作为 MMSegmentation 基础数据集类.

`BaseSegDataset` 继承自 MMEngine 里的 `BaseDataset` 而 `CustomDataset` 继承自 PyTorch 官方 `Dataset`. `CustomDataset` 和 `BaseSegDataset` 的区别在于以下几点:

- `BaseSegDataset` 移除了 `CustomDataset` 里和评估有关的所有方法, 例如 `format_results`, `pre_eval`, `get_gt_seg_map_by_idx`, `get_gt_seg_maps` 和 `evaluate`.
- `BaseSegDataset` 将 `CustomDataset` 里的方法 `load_annotations` 替换成了 `load_data_list`.
- `BaseSegDataset` 将 `CustomDataset` 里的成员变量 `img_infos` 和 `split` 分别替换成了 `data_list` 和 `ann_file`.
- `BaseSegDataset` 将 `CustomDataset` 里的 `CLASSES` and `PALETTE` 整合成 `METAINFO` 字典里的两个域.

# 设计你自己的数据集

在 MMSegmentation 1.x 版本里, 所有提供的数据集的类例如 `CityscapesDataset` 和 `ADE20KDataset` 都继承自 `BaseSegDataset`.
一些常用的方法被定义在 `BaseSegDataset` 里面, 例如加载数据信息, 包括元信息如数据集注释的类别和调色盘的值, 以及数据信息如图像和注释的路径.

如果你想设计你自己的数据集的类, 你可以仿照我们提供的数据集的类并增加你自己的定制化方法. 关于数据集的类的更多信息可以参考 [MMEngine 里的教程](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/basedataset.md).
