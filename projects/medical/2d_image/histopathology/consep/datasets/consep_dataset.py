from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class ConsepDataset(BaseSegDataset):
    """ConsepDataset dataset.

    In segmentation map annotation for ConsepDataset,
    ``reduce_zero_label`` is fixed to False. The ``img_suffix``
    is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png'.

    Args:
        img_suffix (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
    """
    METAINFO = dict(
        classes=('background', 'other', 'inflammatory', 'healthy epithelial',
                 'dysplastic/malignant epithelial', 'fibroblast', 'muscle',
                 'endothelial'))

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)
