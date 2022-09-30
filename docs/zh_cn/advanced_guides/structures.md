# 数据结构

为了统一模型和各功能模块之间的输入和输出的接口, 在 OpenMMLab 2.0 MMEngine 中定义了一套抽象数据结构, 实现了基础的增/删/查/改功能, 支持不同设备间的数据迁移, 也支持了如
`.cpu()`, `.cuda()`, `.get()` 和 `.detach()` 的类字典和张量的操作。具体可以参考 [MMEngine 文档](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/data_element.md)。

同样的, MMSegmentation 亦遵循了 OpenMMLab 2.0 各模块间的接口协议, 定义了 `SegDataSample` 用来封装语义分割任务所需要的数据。

## 语义分割数据 SegDataSample

[SegDataSample](mmseg.structures.SegDataSample) 包括了三个主要数据字段 `gt_sem_seg`, `pred_sem_seg` 和 `seg_logits`, 分别用来存放标注信息, 预测结果和预测的未归一化前的 logits 值。

| 字段           | 类型                      | 描述                            |
| -------------- | ------------------------- | ------------------------------- |
| gt_sem_seg     | [`PixelData`](#pixeldata) | 图像标注信息.                   |
| pred_instances | [`PixelData`](#pixeldata) | 图像预测结果.                   |
| seg_logits     | [`PixelData`](#pixeldata) | 模型预测未归一化前的 logits 值. |

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
assert 'data' in data_sample.gt_sem_seg
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
