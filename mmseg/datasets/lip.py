# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LIPDataset(BaseSegDataset):
    """LIP dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('Background', 'Hat', 'Hair', 'Glove', 'Sunglasses',
                 'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
                 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
                 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe',
                 'Right-shoe'),
        palette=(
            [0, 0, 0],
            [128, 0, 0],
            [255, 0, 0],
            [0, 85, 0],
            [170, 0, 51],
            [255, 85, 0],
            [0, 0, 85],
            [0, 119, 221],
            [85, 85, 0],
            [0, 85, 85],
            [85, 51, 0],
            [52, 86, 128],
            [0, 128, 0],
            [0, 0, 255],
            [51, 170, 221],
            [0, 255, 255],
            [85, 255, 170],
            [170, 255, 85],
            [255, 255, 0],
            [255, 170, 0],
        ))

    def __init__(self, **kwargs) -> None:
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
