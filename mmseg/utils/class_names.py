# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import is_str


def cityscapes_classes():
    """Cityscapes class names for external use."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def ade_classes():
    """ADE20K class names for external use."""
    return [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag'
    ]


def voc_classes():
    """Pascal VOC class names for external use."""
    return [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor'
    ]


def pcontext_classes():
    """Pascal Context class names for external use."""
    return [
        'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird',
        'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat',
        'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain',
        'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
        'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse',
        'person', 'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
        'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track',
        'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window',
        'wood'
    ]


def cocostuff_classes():
    """CocoStuff class names for external use."""
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
        'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
        'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
        'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
        'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood'
    ]


def loveda_classes():
    """LoveDA class names for external use."""
    return [
        'background', 'building', 'road', 'water', 'barren', 'forest',
        'agricultural'
    ]


def potsdam_classes():
    """Potsdam class names for external use."""
    return [
        'impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
        'clutter'
    ]


def vaihingen_classes():
    """Vaihingen class names for external use."""
    return [
        'impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
        'clutter'
    ]


def isaid_classes():
    """iSAID class names for external use."""
    return [
        'background', 'ship', 'store_tank', 'baseball_diamond', 'tennis_court',
        'basketball_court', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle',
        'Small_Vehicle', 'Helicopter', 'Swimming_pool', 'Roundabout',
        'Soccer_ball_field', 'plane', 'Harbor'
    ]


def stare_classes():
    """stare class names for external use."""
    return ['background', 'vessel']


def mapillary_v1_classes():
    """mapillary_v1 class names for external use."""
    return [
        'Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
        'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking',
        'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane', 'Sidewalk',
        'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist',
        'Other Rider', 'Lane Marking - Crosswalk', 'Lane Marking - General',
        'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water',
        'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin',
        'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole',
        'Phone Booth', 'Pothole', 'Street Light', 'Pole', 'Traffic Sign Frame',
        'Utility Pole', 'Traffic Light', 'Traffic Sign (Back)',
        'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car',
        'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer',
        'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled'
    ]


def mapillary_v1_palette():
    """mapillary_v1_ palette for external use."""
    return [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
            [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
            [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
            [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
            [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
            [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
            [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
            [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
            [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
            [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
            [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
            [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
            [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
            [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142],
            [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
            [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]


def mapillary_v2_classes():
    """mapillary_v2 class names for external use."""
    return [
        'Bird', 'Ground Animal', 'Ambiguous Barrier', 'Concrete Block', 'Curb',
        'Fence', 'Guard Rail', 'Barrier', 'Road Median', 'Road Side',
        'Lane Separator', 'Temporary Barrier', 'Wall', 'Bike Lane',
        'Crosswalk - Plain', 'Curb Cut', 'Driveway', 'Parking',
        'Parking Aisle', 'Pedestrian Area', 'Rail Track', 'Road',
        'Road Shoulder', 'Service Lane', 'Sidewalk', 'Traffic Island',
        'Bridge', 'Building', 'Garage', 'Tunnel', 'Person', 'Person Group',
        'Bicyclist', 'Motorcyclist', 'Other Rider',
        'Lane Marking - Dashed Line', 'Lane Marking - Straight Line',
        'Lane Marking - Zigzag Line', 'Lane Marking - Ambiguous',
        'Lane Marking - Arrow (Left)', 'Lane Marking - Arrow (Other)',
        'Lane Marking - Arrow (Right)',
        'Lane Marking - Arrow (Split Left or Straight)',
        'Lane Marking - Arrow (Split Right or Straight)',
        'Lane Marking - Arrow (Straight)', 'Lane Marking - Crosswalk',
        'Lane Marking - Give Way (Row)', 'Lane Marking - Give Way (Single)',
        'Lane Marking - Hatched (Chevron)',
        'Lane Marking - Hatched (Diagonal)', 'Lane Marking - Other',
        'Lane Marking - Stop Line', 'Lane Marking - Symbol (Bicycle)',
        'Lane Marking - Symbol (Other)', 'Lane Marking - Text',
        'Lane Marking (only) - Dashed Line', 'Lane Marking (only) - Crosswalk',
        'Lane Marking (only) - Other', 'Lane Marking (only) - Test',
        'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water',
        'Banner', 'Bench', 'Bike Rack', 'Catch Basin', 'CCTV Camera',
        'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Parking Meter',
        'Phone Booth', 'Pothole', 'Signage - Advertisement',
        'Signage - Ambiguous', 'Signage - Back', 'Signage - Information',
        'Signage - Other', 'Signage - Store', 'Street Light', 'Pole',
        'Pole Group', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Cone',
        'Traffic Light - General (Single)', 'Traffic Light - Pedestrians',
        'Traffic Light - General (Upright)',
        'Traffic Light - General (Horizontal)', 'Traffic Light - Cyclists',
        'Traffic Light - Other', 'Traffic Sign - Ambiguous',
        'Traffic Sign (Back)', 'Traffic Sign - Direction (Back)',
        'Traffic Sign - Direction (Front)', 'Traffic Sign (Front)',
        'Traffic Sign - Parking', 'Traffic Sign - Temporary (Back)',
        'Traffic Sign - Temporary (Front)', 'Trash Can', 'Bicycle', 'Boat',
        'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle',
        'Trailer', 'Truck', 'Vehicle Group', 'Wheeled Slow', 'Water Valve',
        'Car Mount', 'Dynamic', 'Ego Vehicle', 'Ground', 'Static', 'Unlabeled'
    ]


def mapillary_v2_palette():
    """mapillary_v2_ palette for external use."""
    return [[165, 42, 42], [0, 192, 0], [250, 170, 31], [250, 170, 32],
            [196, 196, 196], [190, 153, 153], [180, 165, 180], [90, 120, 150],
            [250, 170, 33], [250, 170, 34], [128, 128, 128], [250, 170, 35],
            [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170],
            [250, 170, 36], [250, 170, 160], [250, 170, 37], [96, 96, 96],
            [230, 150, 140], [128, 64, 128], [110, 110, 110], [110, 110, 110],
            [244, 35, 232], [128, 196, 128], [150, 100, 100], [70, 70, 70],
            [150, 150, 150], [150, 120, 90], [220, 20, 60], [220, 20, 60],
            [255, 0, 0], [255, 0, 100], [255, 0, 200], [255, 255, 255],
            [255, 255, 255], [250, 170, 29], [250, 170, 28], [250, 170, 26],
            [250, 170, 25], [250, 170, 24], [250, 170, 22], [250, 170, 21],
            [250, 170, 20], [255, 255, 255], [250, 170, 19], [250, 170, 18],
            [250, 170, 12], [250, 170, 11], [255, 255, 255], [255, 255, 255],
            [250, 170, 16], [250, 170, 15], [250, 170, 15], [255, 255, 255],
            [255, 255, 255], [255, 255, 255], [255, 255, 255], [64, 170, 64],
            [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152],
            [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30],
            [100, 140, 180], [220, 128, 128], [222, 40, 40], [100, 170, 30],
            [40, 40, 40], [33, 33, 33], [100, 128, 160], [20, 20, 255],
            [142, 0, 0], [70, 100, 150], [250, 171, 30], [250, 172, 30],
            [250, 173, 30], [250, 174, 30], [250, 175, 30], [250, 176, 30],
            [210, 170, 100], [153, 153, 153], [153, 153, 153], [128, 128, 128],
            [0, 0, 80], [210, 60, 60], [250, 170, 30], [250, 170, 30],
            [250, 170, 30], [250, 170, 30], [250, 170, 30], [250, 170, 30],
            [192, 192, 192], [192, 192, 192], [192, 192, 192], [220, 220, 0],
            [220, 220, 0], [0, 0, 196], [192, 192, 192], [220, 220, 0],
            [140, 140, 20], [119, 11, 32], [150, 0, 255], [0, 60, 100],
            [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64],
            [0, 0, 110], [0, 0, 70], [0, 0, 142], [0, 0, 192], [170, 170, 170],
            [32, 32, 32], [111, 74, 0], [120, 10, 10], [81, 0, 81],
            [111, 111, 0], [0, 0, 0]]


def cityscapes_palette():
    """Cityscapes palette for external use."""
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]


def ade_palette():
    """ADE20K palette for external use."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
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
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


def voc_palette():
    """Pascal VOC palette for external use."""
    return [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def pcontext_palette():
    """Pascal Context palette for external use."""
    return [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
            [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
            [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
            [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140],
            [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
            [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
            [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
            [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
            [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8],
            [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8],
            [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
            [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
            [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
            [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0],
            [0, 235, 255], [0, 173, 255], [31, 0, 255]]


def cocostuff_palette():
    """CocoStuff palette for external use."""
    return [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
            [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
            [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
            [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
            [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
            [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
            [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160], [0, 32, 0],
            [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0],
            [192, 128, 32], [128, 96, 128], [0, 0, 128], [64, 0, 32],
            [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
            [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64],
            [192, 0, 32], [128, 96, 0], [128, 0, 192], [0, 128, 32],
            [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
            [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64],
            [128, 128, 32], [192, 32, 128], [0, 64, 192], [0, 0, 32],
            [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128],
            [128, 192, 192], [0, 0, 160], [192, 160, 128], [128, 192, 0],
            [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
            [64, 160, 0], [0, 64, 0], [192, 128, 224], [64, 32, 0],
            [0, 192, 128], [64, 128, 224], [192, 160, 0], [0, 192, 0],
            [192, 128, 96], [192, 96, 128], [0, 64, 128], [64, 0, 96],
            [64, 224, 128], [128, 64, 0], [192, 0, 224], [64, 96, 128],
            [128, 192, 128], [64, 0, 224], [192, 224, 128], [128, 192, 64],
            [192, 0, 96], [192, 96, 0], [128, 64, 192], [0, 128, 96],
            [0, 224, 0], [64, 64, 64], [128, 128, 224], [0, 96, 0],
            [64, 192, 192], [0, 128, 224], [128, 224, 0], [64, 192, 64],
            [128, 128, 96], [128, 32, 128], [64, 0, 192], [0, 64, 96],
            [0, 160, 128], [192, 0, 64], [128, 64, 224], [0, 32, 128],
            [192, 128, 192], [0, 64, 224], [128, 160, 128], [192, 128, 0],
            [128, 64, 32], [128, 32, 64], [192, 0, 128], [64, 192, 32],
            [0, 160, 64], [64, 0, 0], [192, 192, 160], [0, 32, 64],
            [64, 128, 128], [64, 192, 160], [128, 160, 64], [64, 128, 0],
            [192, 192, 32], [128, 96, 192], [64, 0, 128], [64, 64, 32],
            [0, 224, 192], [192, 0, 0], [192, 64, 160], [0, 96, 192],
            [192, 128, 128], [64, 64, 160], [128, 224, 192], [192, 128, 64],
            [192, 64, 32], [128, 96, 64], [192, 0, 192], [0, 192, 32],
            [64, 224, 64], [64, 0, 64], [128, 192, 160], [64, 96, 64],
            [64, 128, 192], [0, 192, 160], [192, 224, 64], [64, 128, 64],
            [128, 192, 32], [192, 32, 192], [64, 64, 192], [0, 64, 32],
            [64, 160, 192], [192, 64, 64], [128, 64, 160], [64, 32, 192],
            [192, 192, 192], [0, 64, 160], [192, 160, 192], [192, 192, 0],
            [128, 64, 96], [192, 32, 64], [192, 64, 128], [64, 192, 96],
            [64, 160, 64], [64, 64, 0]]


def loveda_palette():
    """LoveDA palette for external use."""
    return [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
            [159, 129, 183], [0, 255, 0], [255, 195, 128]]


def potsdam_palette():
    """Potsdam palette for external use."""
    return [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
            [255, 255, 0], [255, 0, 0]]


def vaihingen_palette():
    """Vaihingen palette for external use."""
    return [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
            [255, 255, 0], [255, 0, 0]]


def isaid_palette():
    """iSAID palette for external use."""
    return [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
            [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127,
                                                       127], [0, 0, 127],
            [0, 0, 191], [0, 0, 255], [0, 191, 127], [0, 127, 191],
            [0, 127, 255], [0, 100, 155]]


def stare_palette():
    """STARE palette for external use."""
    return [[120, 120, 120], [6, 230, 230]]


def synapse_palette():
    """Synapse palette for external use."""
    return [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255],
            [255, 0, 255], [255, 255, 0], [60, 255, 255], [240, 240, 240]]


def synapse_classes():
    """Synapse class names for external use."""
    return [
        'background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
        'liver', 'pancreas', 'spleen', 'stomach'
    ]


def lip_classes():
    """LIP class names for external use."""
    return [
        'background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
        'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
        'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
        'rightShoe'
    ]


def lip_palette():
    """LIP palette for external use."""
    return [
        'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'UpperClothes',
        'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
        'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe',
        'Right-shoe'
    ]


def bdd100k_classes():
    """BDD100K class names for external use(the class name is compatible with
    Cityscapes )."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def bdd100k_palette():
    """bdd100k palette for external use(same with cityscapes)"""
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]


def hsidrive_classes():
    """HSI Drive 2.0 class names for external use."""
    return [
        'unlabelled', 'road', 'road marks', 'vegetation', 'painted metal',
        'sky', 'concrete', 'pedestrian', 'water', 'unpainted metal', 'glass'
    ]


def hsidrive_palette():
    """HSI Drive 2.0 palette for external use."""
    return [[0, 0, 0], [77, 77, 77], [255, 255, 255], [0, 255, 0], [255, 0, 0],
            [0, 0, 255], [102, 51, 0], [255, 255, 0], [0, 207, 250],
            [255, 166, 0], [0, 204, 204]]


dataset_aliases = {
    'cityscapes': ['cityscapes'],
    'ade': ['ade', 'ade20k'],
    'voc': ['voc', 'pascal_voc', 'voc12', 'voc12aug'],
    'pcontext': ['pcontext', 'pascal_context', 'voc2010'],
    'loveda': ['loveda'],
    'potsdam': ['potsdam'],
    'vaihingen': ['vaihingen'],
    'cocostuff': [
        'cocostuff', 'cocostuff10k', 'cocostuff164k', 'coco-stuff',
        'coco-stuff10k', 'coco-stuff164k', 'coco_stuff', 'coco_stuff10k',
        'coco_stuff164k'
    ],
    'isaid': ['isaid', 'iSAID'],
    'stare': ['stare', 'STARE'],
    'lip': ['LIP', 'lip'],
    'mapillary_v1': ['mapillary_v1'],
    'mapillary_v2': ['mapillary_v2'],
    'bdd100k': ['bdd100k'],
    'hsidrive': [
        'hsidrive', 'HSIDrive', 'HSI-Drive', 'hsidrive20', 'HSIDrive20',
        'HSI-Drive20'
    ]
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
