# 数据结构

在模型的训练/测试过程中，组件之间往往有大量的数据需要传递，不同的任务或算法传递的数据通常是不一样的。
这严重影响了算法库的拓展性及复用性。 因此，为了解决上述问题，我们基于 [DataElement tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/data_element.md)
将各任务所需的数据统一封装入 `data_sample` 中。 MMEngine 的抽象数据接口实现了基础的增/删/改/查功能，支持不同设备间的数据迁移，也支持了如 `.cpu()`, `.cuda()`, `.get()` and `.detach()` 的类字典和张量的操作，
充分满足了数据的日常使用需求，这也使得不同算法的接口可以统一为以下形式：

```python
for img, data_sample in dataloader:
  seg_logit = model(img, data_sample)
```

得益于统一的数据封装，算法库内的 [`visualizer`](./visualizers.md), [`evaluator`](./evaluation.md), [`dataset`](./datasets.md) 等各个模块间的数据流通都得到了极大的简化。在 MMSegmentation 中，我们对数据接口类型作出以下约定：

- **xxxData**: 单一粒度的数据标注或模型输出。目前 MMEngine 内置了三种粒度的[`数据元素(data element)`](https://github.com/open-mmlab/mmengine/tree/main/mmengine/structures)，包括实例级数据（`InstanceData`），像素级数据（`PixelData`）以及图像级的标签数据（`LabelData`）。
  目前 MMSegmentation 只支持语义分割任务，所以只使用 `PixelData` 来封装图像和对应的标注。
- **xxxDataSample**: 继承自 [`BaseDataElement`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/base_data_element.py)，用于保存单个任务的训练或测试样本的**所有**标注及预测信息。
  在 MMSegmentation 中，我们基于现在支持的语义分割任务及其所需要的数据封装了一个数据抽象 [`SegDataSample`](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py)，用于保存语义分割任务的训练或测试样本的所有标注及预测信息。

总的来说， `BaseDataElement` 包括两类数据，一类是 `data`， 里面包含多种 ground truth 例如边界框，实例掩码和语义掩码，另一类是 `metainfo`，里面包括数据集的元信息，例如图像形状大小 `img_shape`，图像的编号 `img_id` 等确保数据集完整性的信息。当创建新的　`BaseDataElement` 时，用户应该对这两类属性做出明确的声明和区分。

## 语义分割 PixelData

在**语义分割**任务中，模型关注的是像素级别的图像样本，因此我们使用 `PixelData` 来封装该任务所需的数据。其所需的训练标注和预测输出通常包含了像素级别的图像和对应的标注。以下代码示例展示了如何使用 `PixelData` 数据抽象接口来封装语义分割任务中使用的数据类型。

```python
import torch
from mmengine.structures import PixelData

img_meta = dict(img_shape=(4, 4, 3),
                 pad_shape=(4, 4, 3))
# 定义 gt_segmentations 用于封装模型的输出信息
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
```

MMSegmentation 中对 `PixelData` 字段的约定如下表所示:

| Field | Type           | Description     |
| ----- | -------------- | --------------- |
| data  | `torch.Tensor` | 图像标注的数据. |

因为语义分割模型通常只输出每个像素的分类结果，所以我们只需要确保每个像素被分到对应的类别中。

## 语义分割数据抽象 SegDataSample

[SegDataSample](mmseg.structures.SegDataSample) 被用来封装语义分割任务所需要的数据。 它包括了三个主要数据字段 `gt_sem_seg`, `pred_sem_seg` 和 `seg_logits`, 分别用来存放标注信息和预测结果和预测 logits 值。

| Field          | Type                      | Description           |
| -------------- | ------------------------- | --------------------- |
| gt_sem_seg     | [`PixelData`](#pixeldata) | 图像标注信息.         |
| pred_instances | [`PixelData`](#pixeldata) | 图像预测结果.         |
| seg_logits     | [`PixelData`](#pixeldata) | 图像预测的 logits 值. |

以下示例代码展示了 `SegDataSample` 的使用方法：

```python
import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

img_meta = dict(img_shape=(4, 4, 3),
                 pad_shape=(4, 4, 3))
data_sample = SegDataSample()
# 定义 gt_segmentations 用于封装模型的输出信息
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))

# 增加和处理 SegDataSample　中的属性
data_sample.gt_sem_seg = gt_segmentations
assert 'gt_sem_seg' in data_sample
assert 'sem_seg' in data_sample.gt_sem_seg
assert 'img_shape' in data_sample.gt_sem_seg.metainfo_keys()
print(data_sample.gt_sem_seg.shape)
'''
(4, 4)
'''
print(data_sample)
'''
<SegDataSample(

    META INFORMATION

    DATA FIELDS
    gt_sem_seg: <PixelData(

            META INFORMATION
            img_shape: (4, 4, 3)
            pad_shape: (4, 4, 3)

            DATA FIELDS
            data: tensor([[[1, 1, 1, 0],
                         [1, 0, 1, 1],
                         [1, 1, 1, 1],
                         [0, 1, 0, 1]]])
        ) at 0x1c2b4156460>
) at 0x1c2aae44d60>
'''

# 删除和修改 SegDataSample　中的属性
data_sample = SegDataSample()
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
data_sample.gt_sem_seg = gt_segmentations
data_sample.gt_sem_seg.set_metainfo(dict(img_shape=(4,4,9), pad_shape=(4,4,9)))
del data_sample.gt_sem_seg.img_shape

# 类张量的操作
data_sample = SegDataSample()
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
cuda_gt_segmentations = gt_segmentations.cuda()
cuda_gt_segmentations = gt_segmentations.to('cuda:0')
cpu_gt_segmentations = cuda_gt_segmentations.cpu()
cpu_gt_segmentations = cuda_gt_segmentations.to('cpu')
```

## 在 SegDataSample 中自定义新的属性

如果你想在 `SegDataSample` 中自定义新的属性，你可以参考下面的 [SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py) 示例:

```python
class SegDataSample(BaseDataElement):
    ...

    @property
    def xxx_property(self) -> xxxData:
        return self._xxx_property

    @xxx_property.setter
    def xxx_property(self, value: xxxData) -> None:
        self.set_field(value, '_xxx_property', dtype=xxxData)

    @xxx_property.deleter
    def xxx_property(self) -> None:
        del self._xxx_property
```

这样一个新的属性 `xxx_property` 就将被增加到 `SegDataSample` 里面了。
