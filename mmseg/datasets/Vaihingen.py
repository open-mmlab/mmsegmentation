from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class VaihingenDataset(CustomDataset):
    """ISPRS_2d_semantic_labeling_Vaihingen dataset.

    In segmentation map annotation for Vaihingen, 0 stands for the sixth class: Clutter/background
    ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.tif' and ``seg_map_suffix`` is fixed to
    '.tif'.
    """
    CLASSES = (
        'imp surf', 'building', 'low_veg', 'tree', 'car','clutter')

    PALETTE = [[255, 255, 255], [0, 0, 255], 
    [0, 255, 255],[0, 255, 0], [255, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(VaihingenDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='_noBoundary.png',
            reduce_zero_label=True,
            **kwargs)
