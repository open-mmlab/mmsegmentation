import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DRIVEDataset(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '_manual1.jpg'.
    """

    CLASSES = ('background', 'vessel')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(DRIVEDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_manual1.jpg',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
