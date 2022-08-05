# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ChaseDB1Dataset(BaseSegDataset):
    """Chase_db1 dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.
    """
    METAINFO = dict(
        classes=('background', 'vessel'),
        palette=[[120, 120, 120], [6, 230, 230]])

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='_1stHO.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.data_prefix['img_path'])
