# Structures

In the training/testing process of a model, there is often a large amount of data to be passed between modules, and the data required by different tasks or algorithms is usually different.
In order to solve the above problem, we use [DataElement tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/data_element.md) from MMEngine to encapsulate the data required for each task into `data_sample`.
The base class has implemented basic functions of `Create`, `Read`, `Update`, `Delete` and tensor-like and dictionary-like operations such as `.cpu()`, `.cuda()`, `.get()` and `.detach()`. Finally, it could be used like below:

```python
for img, data_sample in dataloader:
  seg_logit = model(img, data_sample)
```

Thanks to the unified data structures, the data flow between each module in the algorithm libraries, such as [`visualizer`](./visualizers.md), [`evaluator`](./evaluation.md), [`dataset`](./datasets.md), is greatly simplified.
In MMSegmentation, we make the following conventions for different data types.

- **xxxData**: Single granularity data annotation or model output. Currently MMEngine has three built-in granularities of [`xxx_data`](https://github.com/open-mmlab/mmengine/tree/main/mmengine/structures), including instance-level data (`InstanceData`), pixel-level data (`PixelData`) and image-level label data (`LabelData`).
  MMSegmentation currently only supports semantic segmentation tasks, thus it only uses `PixelData`.
- **xxxDataSample**: inherited from [`BaseDataElement`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/base_data_element.py), used to hold **all** annotation and prediction information that required by a single task.
  In MMSegmentation, we have encapsulated the semantic segmentation task data abstractions: [`SegDataSample`](https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/structures/seg_data_sample.py).

In general, `BaseDataElement` has two types of data, one is `data` type which includes various types of ground truth such as bounding boxes, instance masks and semantic masks, the other is `metainfo` type which includes meta information of dataset such as `img_shape` and `img_id` to ensure　completeness of data. When creating the new `BaseDataElement`, users should make explicit claims and discrimination on these two types of properties.

In the following, we will introduce the practical application of data elements **Semantic Segmentation PixelData** and data samples Semantic Segmentation Data Abstraction **SegDataSample** in MMSegmentation, respectively.

## Semantic Segmentation PixelData

In the semantic segmentation task, the model concentrates on pixel-level image samples, so we use `PixelData` to encapsulate the data needed for this task. Typically, its required training annotation and prediction output contains pixel-level labels. The following code example shows how to use the `PixelData` data abstraction interface to encapsulate the data types used in the semantic segmentation task.

```python
import torch
from mmengine.structures import PixelData

img_meta = dict(img_shape=(4, 4, 3),
                 pad_shape=(4, 4, 3))
# defining gt_segmentations for encapsulate the ground truth data
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
```

The fields of [`PixelData`](#pixeldata) that will be used are:

| Field | Type           | Description     |
| ----- | -------------- | --------------- |
| data  | `torch.Tensor` | data of images. |

Since semantic segmentation models usually only output one pixel-level classification result, we only need to make sure that each pixel is assigned a certain value.

## Semantic Segmentation Data Abstraction SegDataSample

[SegDataSample](mmseg.structures.SegDataSample) is used to encapsulate the data needed for the semantic segmentation task. It contains three main fields `gt_sem_seg`, `pred_sem_seg` and `seg_logits`, which are used to store the annotation information and prediction results respectively.

| Field          | Type                      | Description                     |
| -------------- | ------------------------- | ------------------------------- |
| gt_sem_seg     | [`PixelData`](#pixeldata) | Annotation information.         |
| pred_instances | [`PixelData`](#pixeldata) | The predicted result.           |
| seg_logits     | [`PixelData`](#pixeldata) | The logits of predicted result. |

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
