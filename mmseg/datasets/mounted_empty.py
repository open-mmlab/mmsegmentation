from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class MountedEmpty(BaseSegDataset):

    METAINFO = dict(
        classes=('full','empty',),
        palette=([0,0,0],[0,128,0],),
    )

    def __init__(self,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

