# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SynapseDataset(BaseSegDataset):
    """Synapse dataset.

    Before dataset preprocess of Synapse, there are total 13 categories of
    foreground which does not include background. After preprocessing, 8
    foreground categories are kept while the other 5 foreground categories are
    handled as background. The ``img_suffix`` is fixed to '.jpg' and
    ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'aorta', 'gallbladder', 'left_kidney',
                 'right_kidney', 'liver', 'pancreas', 'spleen', 'stomach'),
        palette=[[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0],
                 [0, 255, 255], [255, 0, 255], [255, 255, 0], [60, 255, 255],
                 [240, 240, 240]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
