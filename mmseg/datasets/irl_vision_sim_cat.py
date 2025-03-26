from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class IRLVisionSimCATDataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
                    '_background_', 
                    'apple', 'banana', 'book', 'bowl', 
                    'can', 'cap', 'cereal', 'cup', 
                    'flashlight', 'fork', 'juice_box', 'keyboard', 
                    'knife', 'laptop', 'lemon', 'marker', 
                    'milk', 'monitor', 'mouse', 'mug', 
                    'orange', 'peach', 'pear', 'pen', 
                    'plate', 'pringles', 'scissors', 'sponge', 
                    'spoon'

                ),
        palette=[
                [0, 0, 0], 
                [139, 152, 218], [37, 121, 67], [238, 44, 243], [154, 38, 123], 
                [209, 169, 129], [23, 86, 101], [171, 37, 22], [243, 122, 46], 
                [29, 30, 220], [46, 36, 59], [221, 90, 231], [0, 255, 33], 
                [238, 1, 11], [214, 135, 52], [107, 225, 139], [10, 17, 218], 
                [189, 8, 9], [214, 151, 117], [240, 238, 119], [247, 91, 58], 
                [188, 243, 59], [247, 126, 168], [136, 52, 244], [213, 55, 234], 
                [83, 193, 216], [105, 135, 229], [30, 131, 132], [72, 125, 9], 
                [216, 81, 143]
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
    
 
 
 
 
