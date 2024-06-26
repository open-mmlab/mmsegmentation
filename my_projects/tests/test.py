"""A data structure interface of MMSegmentation. They are used as
interfaces between different components.

The attributes in ``SegDataSample`` are divided into several parts:

    - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
    - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
    - ``seg_logits``(PixelData): Predicted logits of semantic segmentation.

Examples:
        
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

    >>> data_sample = SegDataSample()
    >>> gt_sem_seg_data = dict(sem_seg=torch.rand(1, 4, 4))
    >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
    >>> data_sample.gt_sem_seg = gt_sem_seg
    >>> assert 'gt_sem_seg' in data_sample
    >>> assert 'sem_seg' in data_sample.gt_sem_seg
"""
import torch
import numpy as np
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

data_sample = SegDataSample()
img_meta = dict(img_shape=(4, 4, 3),
               pad_shape=(4, 4, 3))
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
data_sample.gt_sem_seg = gt_segmentations
print(data_sample.gt_sem_seg.shape)
print(data_sample)

data_sample = SegDataSample()
gt_sem_seg_data = dict(sem_seg=torch.rand(1, 4, 4))
gt_sem_seg = PixelData(**gt_sem_seg_data)
data_sample.gt_sem_seg = gt_sem_seg
assert 'gt_sem_seg' in data_sample
assert 'sem_seg' in data_sample.gt_sem_seg