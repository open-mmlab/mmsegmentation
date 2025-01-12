from mmseg.utils import (
    hots_v1_classes, 
    hots_v1_palette,
    irl_vision_sim_classes,
    irl_vision_sim_palette,
    ade_classes, 
    ade_palette,
    hots_v1_cat_classes, 
    hots_v1_cat_palette,
    irl_vision_sim_cat_classes,
    irl_vision_sim_cat_palette,
    arid20cat_classes,
    arid20cat_palette,
    arid10cat_classes,
    arid10cat_palette
)


HOTS2HOTS_CAT = {
    0  :  0,
    1  :  1,
    2  :  2,
    3  :  3,
    4  :  3,
    5  :  3,
    6  :  4,
    7  :  5,
    8  :  5,
    9  :  5,
    10  :  5,
    11  :  5,
    12  :  6,
    13  :  6,
    14  :  6,
    15  :  7,
    16  :  7,
    17  :  8,
    18  :  8,
    19  :  8,
    20  :  9,
    21  :  10,
    22  :  11,
    23  :  12,
    24  :  13,
    25  :  13,
    26  :  14,
    27  :  14,
    28  :  15,
    29  :  16,
    30  :  16,
    31  :  17,
    32  :  18,
    33  :  19,
    34  :  20,
    35  :  20,
    36  :  20,
    37  :  21,
    38  :  21,
    39  :  22,
    40  :  22,
    41  :  22,
    42  :  23,
    43  :  23,
    44  :  24,
    45  :  24,
    46  :  25
}

IRL_VISION2IRL_VISION_CAT = {
    0  :  0,
    1  :  1,
    2  :  2,
    3  :  3,
    4  :  3,
    5  :  3,
    6  :  3,
    7  :  3,
    8  :  3,
    9  :  4,
    10  :  4,
    11  :  4,
    12  :  4,
    13  :  4,
    14  :  4,
    15  :  5,
    16  :  5,
    17  :  5,
    18  :  5,
    19  :  5,
    20  :  6,
    21  :  6,
    22  :  6,
    23  :  7,
    24  :  7,
    25  :  7,
    26  :  8,
    27  :  8,
    28  :  8,
    29  :  8,
    30  :  9,
    31  :  9,
    32  :  9,
    33  :  9,
    34  :  10,
    35  :  11,
    36  :  11,
    37  :  11,
    38  :  11,
    39  :  12,
    40  :  13,
    41  :  14,
    42  :  14,
    43  :  15,
    44  :  16,
    45  :  16,
    46  :  16,
    47  :  17,
    48  :  18,
    49  :  19,
    50  :  20,
    51  :  20,
    52  :  20,
    53  :  20,
    54  :  21,
    55  :  22,
    56  :  23,
    57  :  24,
    58  :  24,
    59  :  24,
    60  :  24,
    61  :  25,
    62  :  25,
    63  :  25,
    64  :  26,
    65  :  26,
    66  :  26,
    67  :  27,
    68  :  28,
    69  :  28,
    70  :  28,
    71  :  29
}

IRL_VISION_CAT2HOTS_CAT = {
    0  :  0,
    1  :  1,
    2  :  2,
    3  :  3,
    4  :  4,
    5  :  5,
    8  :  6,
    10  :  7,
    11  :  8,
    12  :  9,
    13  :  10,
    14  :  11,
    15  :  12,
    16  :  13,
    17  :  14,
    18  :  15,
    19  :  16,
    21  :  17,
    22  :  18,
    23  :  19,
    24  :  20,
    25  :  21,
    26  :  22,
    27  :  23,
    29  :  24
}

HOTS_CAT2IRL_VISION_CAT = {
    0  :  0,
    1  :  1,
    2  :  2,
    3  :  3,
    4  :  4,
    5  :  5,
    6  :  8,
    7  :  10,
    8  :  11,
    9  :  12,
    10  :  13,
    11  :  14,
    12  :  15,
    13  :  16,
    14  :  17,
    15  :  18,
    16  :  19,
    17  :  21,
    18  :  22,
    19  :  23,
    20  :  24,
    21  :  25,
    22  :  26,
    23  :  27,
    24  :  29
}

ADE20K2HOTS_CAT_CLASS_NAMES = {
    "box"           :   ["juice_box", "milk"],
    "book"          :   ["book"],
    "computer"      :   [
        "laptop",
        "monitor",
        "mouse",
        "keyboard"
    ], 
    "food"          :   [
        "apple",
        "banana",
        "lemon",
        "orange",
        "peach",
        "pear",
        "pringles"
    ], # solid food
    "screen"        :   [
        "laptop",
        "monitor"
    ],
    "crt screen"    :   [
        "laptop",
        "monitor"
    ],
    "plate"         :   ["plate"],
    "monitor"       :   [
        "laptop",
        "monitor"
    ],
    "glass"         :   ["cup", "bowl"] 
}

ADE20K2HOTS_CAT  = {
    41 : [8, 14],
    67 : [3],
    74 : [11, 15, 16, 9],
    120 : [1, 2, 12, 17, 18, 19, 22],
    130 : [11, 15],
    141 : [11, 15],
    142 : [21],
    143 : [11, 15],
    147 : [6, 4]
}

HOTS_CAT2ADE20K = {
    8 : [41],
    14 : [41],
    3 : [67],
    11 : [74, 130, 141, 143],
    15 : [74, 130, 141, 143],
    16 : [74],
    9 : [74],
    1 : [120],
    2 : [120],
    12 : [120],
    17 : [120],
    18 : [120],
    19 : [120],
    22 : [120],
    21 : [142],
    6 : [147],
    4 : [147]
}

# for sod2ade:
#   cereal is box,
#  flashlight = light



ARID20CAT2HOTS_CAT_CLASS_NAMES = {
    "_background_"  :       "_background_", 
    "apple"         :       "apple", 
    "banana"        :       "banana", 
    "bowl"          :       "bowl", 
    "soda_can"      :       "can", 
    "keyboard"      :       "keyboard", 
    "lemon"         :       "lemon", 
    "marker"        :       "marker", 
    "orange"        :       "orange", 
    "peach"         :       "peach", 
    "pear"          :       "pear", 
    "stapler"       :       "stapler"

}   

ARID20CAT2IRL_VISION_CAT_CLASS_NAMES = {
    "_background_"  :       "_background_", 
    "apple"         :       "apple", 
    "banana"        :       "banana", 
    "bowl"          :       "bowl", 
    "soda_can"      :       "can", 
    "cereal_box"    :       "cereal", 
    "flashlight"    :       "flashlight", 
    "keyboard"      :       "keyboard", 
    "lemon"         :       "lemon", 
    "marker"        :       "marker", 
    "coffee_mug"    :       "mug", 
    "orange"        :       "orange", 
    "peach"         :       "peach", 
    "pear"          :       "pear", 
    "sponge"        :       "sponge"
}
DATASET_CLASSES = {
    "HOTS"          :   hots_v1_classes,
    "HOTS_CAT"      :   hots_v1_cat_classes,
    "IRL_VISION"    :   irl_vision_sim_classes,
    "IRL_VISION_CAT":   irl_vision_sim_cat_classes,
    "ADE20K"        :   ade_classes,
    "ARID20"        :   arid20cat_classes,
    "ARID10"        :   arid10cat_classes
}

def get_auto_convert(dataset_name):
    return {
        idx : idx for idx in range(len(DATASET_CLASSES[dataset_name]()))
    }

SOURCE_TARGET_MAP = {
    "HOTS"                  :   {
            "HOTS_CAT"          :   [HOTS2HOTS_CAT],
            "ADE20K"            :   [HOTS2HOTS_CAT, HOTS_CAT2ADE20K],
            "IRL_VISION_CAT"    :   [HOTS2HOTS_CAT, HOTS_CAT2IRL_VISION_CAT],
            "HOTS"              :   [get_auto_convert(dataset_name="HOTS")]
    },
    "IRL_VISION"            :   {
                "IRL_VISION_CAT"   :   [IRL_VISION2IRL_VISION_CAT],
                "HOTS_CAT"          :   [
                    IRL_VISION2IRL_VISION_CAT, 
                    IRL_VISION_CAT2HOTS_CAT
                ],
                "IRL_VISION"    : [get_auto_convert(dataset_name="IRL_VISION")]
    },
    "ADE20K"    :   {
        "HOTS_CAT"      :   [ADE20K2HOTS_CAT],
        "ADE20K"        :   [get_auto_convert(dataset_name="ADE20K")]
    },
    "IRL_VISION_CAT"    :           {
                "HOTS_CAT"          :   [IRL_VISION_CAT2HOTS_CAT],
                "IRL_VISION_CAT"    :[
                    get_auto_convert(dataset_name="IRL_VISION_CAT")
                    ]
    },
    "HOTS_CAT"          :   {
        "IRL_VISION_CAT"    :   [HOTS_CAT2IRL_VISION_CAT],
        "HOTS_CAT"          : [get_auto_convert(dataset_name="HOTS_CAT")]
    },
    "ARID20"            :   {
        "ARID20"            : [get_auto_convert(dataset_name="ARID20")],
        "ARID10"            : [get_auto_convert(dataset_name="ARID10")]
    },
    "ARID10"            :   {
        "ARID20"            : [get_auto_convert(dataset_name="ARID20")],
        "ARID10"            : [get_auto_convert(dataset_name="ARID10")]
    }
}

MISSING_CLASS_NAMES = {
    "HOTS_CAT"      :       {
        "IRL_VISION_CAT"    :       [
            "cap",
            "cereal",
            "flashlight",
            "mug",
            "sponge"
        ]
    },
    "IRL_VISION_CAT"    :   {
        "HOTS_CAT"      :       [
            "stapler"
        ]
    },
    "HOTS"      :       {
        "IRL_VISION"    :       [
            "cap",
            "cereal",
            "flashlight",
            "mug",
            "sponge"
        ]
    },
    "IRL_VISION"    :   {
        "HOTS"      :       [
            "stapler"
        ]
    }
}

def get_missing_class_names(source_dataset, target_dataset):
    if "HOTS" in source_dataset:
        source_dataset = "HOTS"
    elif "IRL" in source_dataset:
        source_dataset = "IRL_VISION"
    
    if "HOTS" in target_dataset:
        target_dataset = "HOTS"
    elif "IRL" in target_dataset:
        target_dataset = "IRL_VISION"
    if target_dataset == source_dataset:
        return []
    return MISSING_CLASS_NAMES[source_dataset][target_dataset]

DATASET_PALETTE = {
    "HOTS"          :   hots_v1_palette,
    "HOTS_CAT"      :   hots_v1_cat_palette,
    "IRL_VISION"    :   irl_vision_sim_palette,
    "IRL_VISION_CAT":   irl_vision_sim_cat_palette,
    "ADE20K"        :   ade_palette,
    "ARID20"        :   arid20cat_palette,
    "ARID10"        :   arid10cat_palette
}

