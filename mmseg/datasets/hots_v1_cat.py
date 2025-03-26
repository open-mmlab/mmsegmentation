from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class HOTSCATDataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
                    '_background_', 
                    'apple', 'banana', 'book', 'bowl', 
                    'can', 'cup', 'fork', 'juice_box', 
                    'keyboard', 'knife', 'laptop', 'lemon', 
                    'marker', 'milk', 'monitor', 'mouse', 
                    'orange', 'peach', 'pear', 'pen', 
                    'plate', 'pringles', 'scissors', 'spoon', 
                    'stapler'
                ),
        palette=[
            [0, 0, 0], 
            [120, 120, 120], [180, 120, 120], [6, 230, 230], [120, 120, 80], 
            [140, 140, 140], [235, 33, 7], [8, 255, 51], [150, 100, 200], 
            [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], 
            [11, 102, 255], [255, 9, 224], [220, 220, 220], [255, 9, 92], 
            [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], 
            [224, 255, 8], [255, 61, 6], [0, 255, 20], [255, 5, 153], 
            [235, 12, 255]
        ]
    )

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
    
 
 
 
 
