# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.cityscapes import CityscapesDataset


@DATASETS.register_module()
class KITTISTEPDataset(CityscapesDataset):
    """KITTI-STEP dataset."""

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs):
        super(KITTISTEPDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
