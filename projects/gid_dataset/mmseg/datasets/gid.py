# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


# 注册数据集类
@DATASETS.register_module()
class GID_Dataset(BaseSegDataset):
    """Gaofen Image Dataset (GID)

    Dataset paper link:
    https://www.sciencedirect.com/science/article/pii/S0034425719303414
    https://x-ytong.github.io/project/GID.html

    GID  6 classes: others, built-up, farmland, forest, meadow, water

    In this example, select 15 images from GID dataset as training set,
    and select 5 images as validation set.
    The selected images are listed as follows:

    GF2_PMS1__L1A0000647767-MSS1
    GF2_PMS1__L1A0001064454-MSS1
    GF2_PMS1__L1A0001348919-MSS1
    GF2_PMS1__L1A0001680851-MSS1
    GF2_PMS1__L1A0001680853-MSS1
    GF2_PMS1__L1A0001680857-MSS1
    GF2_PMS1__L1A0001757429-MSS1
    GF2_PMS2__L1A0000607681-MSS2
    GF2_PMS2__L1A0000635115-MSS2
    GF2_PMS2__L1A0000658637-MSS2
    GF2_PMS2__L1A0001206072-MSS2
    GF2_PMS2__L1A0001471436-MSS2
    GF2_PMS2__L1A0001642620-MSS2
    GF2_PMS2__L1A0001787089-MSS2
    GF2_PMS2__L1A0001838560-MSS2

    The ``img_suffix`` is fixed to '.tif' and ``seg_map_suffix`` is
    fixed to '.tif' for GID.
    """
    METAINFO = dict(
        classes=('Others', 'Built-up', 'Farmland', 'Forest', 'Meadow',
                 'Water'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255],
                 [255, 255, 0], [0, 0, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
