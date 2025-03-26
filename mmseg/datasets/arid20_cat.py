from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ARID20CATDataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
                    "_background_", "apple", "ball", "banana", 
                    "bell_pepper", "binder", "bowl", "cereal_box", 
                    "coffee_mug", "flashlight", "food_bag", "food_box", 
                    "food_can", "glue_stick", "hand_towel", "instant_noodles", 
                    "keyboard", "kleenex", "lemon", "lime", 
                    "marker", "orange", "peach", "pear", 
                    "potato", "shampoo", "soda_can", "sponge", 
                    "stapler", "tomato", "toothpaste", "_unknown_"
                ),
        palette=[
                [0, 0, 0], [69, 167, 125], [98, 36, 68], [166, 50, 141], 
                [127, 254, 225], [255, 243, 59], [53, 166, 152], [61, 228, 157], 
                [22, 187, 196], [33, 15, 250], [234, 19, 6], [95, 152, 115], 
                [255, 120, 0], [160, 54, 224], [252, 77, 216], [255, 0, 200], 
                [168, 175, 110], [190, 90, 100], [197, 39, 15], [137, 233, 180], 
                [233, 229, 20], [30, 241, 158], [137, 34, 172], [21, 154, 206], 
                [104, 190, 112], [219, 205, 202], [96, 129, 51], [32, 203, 104], 
                [29, 241, 69], [35, 229, 130], [124, 162, 250], [163, 156, 156]
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
    
 
 
 
 
