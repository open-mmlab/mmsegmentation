from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LoveDADataset(CustomDataset):
    """LoveDA dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True.
    The ``img_suffix`` and ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')

    PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
               [159, 129, 183], [0, 255, 0], [255, 195, 128]]

    def __init__(self, **kwargs):
        super(LoveDADataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', reduce_zero_label=True, **kwargs)

