import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class ZeroMouldV1Dataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
            'background',
            'correct-coloured',
            'correct-uncoloured',
            'wrong-uncoloured',
            'idk'
        ),
        palette=[
            [0, 0, 0],
            [255, 0, 0],
            [0, 200, 100],
            [255, 225, 0],
            [0, 0, 255]
        ]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.ome.tiff',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args)# and osp.isfile(self.ann_file)
