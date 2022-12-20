# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class REFUGEDataset(BaseSegDataset):
    """REFUGE dataset.

    In segmentation map annotation for HRF, 0 stands for background, which is not
    included in 2 categories. ``reduce_zero_label`` is fixed to True. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('background', ' Optic Cup','Optic Disc'),
        palette=[[120, 120, 120], [6, 230, 230],[56, 59, 120]])

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
        assert self.file_client.exists(self.data_prefix['img_path'])

