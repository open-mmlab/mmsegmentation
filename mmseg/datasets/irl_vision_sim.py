from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class IRLVisionSimDataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
                    "_background_", 
                    "apple", "banana", "book_1", "book_10", 
                    "book_11", "book_15", "book_20", "book_8", 
                    "bowl_1", "bowl_2", "bowl_blue", "bowl_green", 
                    "bowl_red", "bowl_white", "can_coke", "can_coke_zero", 
                    "can_fanta", "can_pepsi", "can_sprite", "cap_black", 
                    "cap_red", "cap_white", "cereal_box_1", "cereal_box_2", 
                    "cereal_box_3", "cup", "cup_glass", "cup_glass_hex", 
                    "cup_paper", "flashlight_black", "flashlight_blue", 
                    "flashlight_red", 
                    "flashlight_yellow", "fork", "juice_box_blue", 
                    "juice_box_green", 
                    "juice_box_orange", "juice_box_pink", "keyboard", "knife", 
                    "laptop_mac_1", "laptop_pc_2", "lemon", "marker_black", 
                    "marker_blue", "marker_red", "milk_box", "monitor_4", 
                    "mouse", "mug_blue", "mug_green", "mug_red", 
                    "mug_yellow", "orange", "peach", "pear", 
                    "pen_black", "pen_blue", "pen_green", "pen_violet", 
                    "plate", "plate_1", "plate_2", "pringles_green", 
                    "pringles_hot", "pringles_orange", "scissors", 
                    "sponge_green", 
                    "sponge_pink", "sponge_yellow", "spoon"

                ),
        palette=[
                [0, 0, 0], 
                [139, 152, 218], [37, 121, 67], [238, 247, 243], [22, 167, 205], 
                [78, 250, 236], [132, 146, 118], [133, 68, 28], [24, 180, 14], 
                [154, 38, 123], [153, 163, 15], [165, 45, 159], [245, 248, 135], 
                [0, 255, 100], [101, 140, 98], [209, 169, 129], [97, 41, 205], 
                [203, 172, 71], [255, 73, 168], [38, 78, 223], [23, 86, 101], 
                [15, 104, 200], [155, 101, 85], [171, 37, 22], [75, 119, 150], 
                [200, 204, 88], [243, 122, 46], [243, 165, 68], [84, 27, 26], 
                [177, 138, 248], [29, 30, 220], [6, 116, 22], [5, 194, 218], 
                [173, 227, 126], [46, 36, 59], [221, 239, 231], [193, 153, 6], 
                [1, 141, 223], [161, 233, 106], [0, 10, 255], [238, 1, 11], 
                [255, 255, 0], [135, 47, 88], [107, 225, 139], [10, 17, 218], 
                [100, 230, 119], [215, 252, 7], [189, 8, 9], [214, 151, 117], 
                [240, 238, 119], [247, 91, 58], [78, 68, 136], [193, 95, 105], 
                [223, 140, 16], [188, 243, 59], [247, 126, 168], [136, 52, 244], 
                [213, 242, 234], [253, 30, 166], [202, 242, 6], [154, 52, 185], 
                [83, 193, 216], [65, 107, 147], [217, 238, 57], [105, 135, 229], 
                [221, 252, 172], [37, 44, 104], [30, 131, 132], [72, 125, 9], 
                [251, 113, 172], [220, 182, 57], [216, 81, 143]
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
    
 
 
 
 
