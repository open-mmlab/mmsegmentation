# Structures

To unify input and output interfaces  between different models and modules, OpenMMLab 2.0 MMEngine defines an abstract data structure,
it has implemented basic functions of `Create`, `Read`, `Update`, `Delete`, supported data transferring among different types of devices
and tensor-like or dictionary-like operations such as `.cpu()`, `.cuda()`, `.get()` and `.detach()`.
More details can be found [here](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/data_element.md).

MMSegmentation also follows this interface protocol and defines `SegDataSample` which is used to encapsulate the data of semantic segmentation task.

## Semantic Segmentation Data SegDataSample

[SegDataSample](mmseg.structures.SegDataSample) includes three main fields `gt_sem_seg`, `pred_sem_seg` and `seg_logits`, which are used to store the annotation information and prediction results respectively.

| Field          | Type                      | Description                                |
| -------------- | ------------------------- | ------------------------------------------ |
| gt_sem_seg     | [`PixelData`](#pixeldata) | Annotation information.                    |
| pred_instances | [`PixelData`](#pixeldata) | The predicted result.                      |
| seg_logits     | [`PixelData`](#pixeldata) | The raw (non-normalized) predicted result. |

The following sample code demonstrates the use of `SegDataSample`.

```python
import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

img_meta = dict(img_shape=(4, 4, 3),
                 pad_shape=(4, 4, 3))
data_sample = SegDataSample()
# defining gt_segmentations for encapsulate the ground truth data
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))

# add and process property in SegDataSample
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

# delete and change property in SegDataSample
data_sample = SegDataSample()
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
data_sample.gt_sem_seg = gt_segmentations
data_sample.gt_sem_seg.set_metainfo(dict(img_shape=(4,4,9), pad_shape=(4,4,9)))
del data_sample.gt_sem_seg.img_shape

# Tensor-like operations
data_sample = SegDataSample()
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
cuda_gt_segmentations = gt_segmentations.cuda()
cuda_gt_segmentations = gt_segmentations.to('cuda:0')
cpu_gt_segmentations = cuda_gt_segmentations.cpu()
cpu_gt_segmentations = cuda_gt_segmentations.to('cpu')
```

## Customize New Property in SegDataSample

If you want to customize new property in `SegDataSample`, you may follow [SegDataSample](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py) below:

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

Then a new property would be added to `SegDataSample`.
