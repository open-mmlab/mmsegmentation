from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class HOTSDataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
                    "_background_", "apple", "banana", "book_blue", "book_white", "book_yellow",
                    "bowl", "can_cassis", "can_coke", "can_fanta", "can_jumbo",
                    "can_pepsi", "cup_black", "cup_glass", "cup_red", "fork_black",
                    "fork_silver", "juice_box_green", "juice_box_orange", "juice_box_pink",
                    "keyboard", "knife", "laptop", "lemon", "marker_blue", "marker_red", 
                    "milk_big", "milk_small", "monitor", "mouse_black", "mouse_silver",
                    "orange", "peach", "pear", "pen_black", "pen_blue", "pen_red",
                    "plate_big", "plate_wide", "pringles_hot", "pringles_red",
                    "pringles_purple", "scissors_black", "scissors_silver", 
                    "spoon_blue", "spoon_silver", "stapler"
                ),
        palette=[[0,0,0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255]])

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
    
 
 
 
 
