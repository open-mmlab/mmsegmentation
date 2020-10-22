import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ChaseDB1Dataset(CustomDataset):
    """Chase_db1 dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '_1stHO.jpg'.
    """

    CLASSES = ('background', 'vessel')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(ChaseDB1Dataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_1stHO.jpg',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
