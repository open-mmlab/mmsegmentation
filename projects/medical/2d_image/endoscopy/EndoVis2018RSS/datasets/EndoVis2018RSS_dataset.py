from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class EndoVis2018RSSDataset(BaseSegDataset):
    """EndoVis2018RSSDataset dataset.

    In segmentation map annotation for EndoVis2018RSSDataset,
    ``reduce_zero_label`` is fixed to False. The ``img_suffix``
    is fixed to '.png' and ``seg_map_suffix`` is fixed to '.png'.

    Args:
        img_suffix (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
    """
    METAINFO = dict(
        classes=('background', 'instrumentShaft', 'instrumentClasper',
                 'instrumentWrist', 'kidneyParenchyma', 'coveredKidney',
                 'thread', 'clamps', 'suturingNeedle', 'suctionInstrument',
                 'intestine', 'ultrasoundProbe'))

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
