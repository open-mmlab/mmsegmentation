# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SynapseDataset(BaseSegDataset):
    """Synapse dataset.

    In segmentation map annotation for Synapse, 0 stands for background, which
    is not include in 13 categories. The ``img_suffix`` is fixed to '.jpg' and
    ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'spleen', 'right_kidney', 'left_kidney',
                 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta',
                 'inferior_vena_cava', 'portal_vein_and_splenic_vein',
                 'pancreas', 'right_adrenal_gland', 'left_adrenal_gland'),
        palette=[[0, 0, 0], [255, 127, 127], [224, 231, 161], [138, 204, 132],
                 [64, 172, 136], [126, 152, 187], [140, 110, 160],
                 [247, 88, 240], [202, 172, 161], [237, 213, 149],
                 [139, 182, 139], [111, 192, 185], [82, 107, 163],
                 [89, 54, 156]])

    def __init__(self, **kwargs) -> None:
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
