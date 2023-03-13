# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MapillaryDataset_v1(BaseSegDataset):
    """Mapillary Vistas Dataset.

    Dataset paper link:
    http://ieeexplore.ieee.org/document/8237796/

    v1.2 contain 66 object classes.
    (37 instance-specific)

    v2.0 contain 124 object classes.
    (70 instance-specific, 46 stuff, 8 void or crowd).

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png' for Mapillary Vistas Dataset.
    """
    METAINFO = dict(
        classes=('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail',
                 'Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain',
                 'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track',
                 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building',
                 'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist',
                 'Other Rider', 'Lane Marking - Crosswalk',
                 'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow',
                 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench',
                 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
                 'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole',
                 'Phone Booth', 'Pothole', 'Street Light', 'Pole',
                 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
                 'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can',
                 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle',
                 'On Rails', 'Other Vehicle', 'Trailer', 'Truck',
                 'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled'),
        palette=[[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
                 [180, 165, 180], [90, 120, 150], [102, 102, 156],
                 [128, 64, 255], [140, 140, 200], [170, 170, 170],
                 [250, 170, 160], [96, 96, 96],
                 [230, 150, 140], [128, 64, 128], [110, 110, 110],
                 [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90],
                 [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200],
                 [200, 128, 128], [255, 255, 255], [64, 170,
                                                    64], [230, 160, 50],
                 [70, 130, 180], [190, 255, 255], [152, 251, 152],
                 [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30],
                 [100, 140, 180], [220, 220, 220], [220, 128, 128],
                 [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33],
                 [100, 128, 160], [142, 0, 0], [70, 100, 150], [210, 170, 100],
                 [153, 153, 153], [128, 128, 128], [0, 0, 80], [250, 170, 30],
                 [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32],
                 [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90],
                 [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
                 [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10,
                                                         10], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class MapillaryDataset_v2(BaseSegDataset):
    """Mapillary Vistas Dataset.

    Dataset paper link:
    http://ieeexplore.ieee.org/document/8237796/

    v1.2 contain 66 object classes.
    (37 instance-specific)

    v2.0 contain 124 object classes.
    (70 instance-specific, 46 stuff, 8 void or crowd).

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png' for Mapillary Vistas Dataset.
    """
    METAINFO = dict(
        classes=(
            'Bird', 'Ground Animal', 'Ambiguous Barrier', 'Concrete Block',
            'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Road Median',
            'Road Side', 'Lane Separator', 'Temporary Barrier', 'Wall',
            'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Driveway',
            'Parking', 'Parking Aisle', 'Pedestrian Area', 'Rail Track',
            'Road', 'Road Shoulder', 'Service Lane', 'Sidewalk',
            'Traffic Island', 'Bridge', 'Building', 'Garage', 'Tunnel',
            'Person', 'Person Group', 'Bicyclist', 'Motorcyclist',
            'Other Rider', 'Lane Marking - Dashed Line',
            'Lane Marking - Straight Line', 'Lane Marking - Zigzag Line',
            'Lane Marking - Ambiguous', 'Lane Marking - Arrow (Left)',
            'Lane Marking - Arrow (Other)', 'Lane Marking - Arrow (Right)',
            'Lane Marking - Arrow (Split Left or Straight)',
            'Lane Marking - Arrow (Split Right or Straight)',
            'Lane Marking - Arrow (Straight)', 'Lane Marking - Crosswalk',
            'Lane Marking - Give Way (Row)',
            'Lane Marking - Give Way (Single)',
            'Lane Marking - Hatched (Chevron)',
            'Lane Marking - Hatched (Diagonal)', 'Lane Marking - Other',
            'Lane Marking - Stop Line', 'Lane Marking - Symbol (Bicycle)',
            'Lane Marking - Symbol (Other)', 'Lane Marking - Text',
            'Lane Marking (only) - Dashed Line',
            'Lane Marking (only) - Crosswalk', 'Lane Marking (only) - Other',
            'Lane Marking (only) - Test', 'Mountain', 'Sand', 'Sky', 'Snow',
            'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack',
            'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box',
            'Mailbox', 'Manhole', 'Parking Meter', 'Phone Booth', 'Pothole',
            'Signage - Advertisement', 'Signage - Ambiguous', 'Signage - Back',
            'Signage - Information', 'Signage - Other', 'Signage - Store',
            'Street Light', 'Pole', 'Pole Group', 'Traffic Sign Frame',
            'Utility Pole', 'Traffic Cone', 'Traffic Light - General (Single)',
            'Traffic Light - Pedestrians', 'Traffic Light - General (Upright)',
            'Traffic Light - General (Horizontal)', 'Traffic Light - Cyclists',
            'Traffic Light - Other', 'Traffic Sign - Ambiguous',
            'Traffic Sign (Back)', 'Traffic Sign - Direction (Back)',
            'Traffic Sign - Direction (Front)', 'Traffic Sign (Front)',
            'Traffic Sign - Parking', 'Traffic Sign - Temporary (Back)',
            'Traffic Sign - Temporary (Front)', 'Trash Can', 'Bicycle', 'Boat',
            'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle',
            'Trailer', 'Truck', 'Vehicle Group', 'Wheeled Slow', 'Water Valve',
            'Car Mount', 'Dynamic', 'Ego Vehicle', 'Ground', 'Static',
            'Unlabeled'),
        palette=[[165, 42, 42], [0, 192, 0], [250, 170, 31], [250, 170, 32],
                 [196, 196, 196], [190, 153, 153], [180, 165, 180],
                 [90, 120, 150], [250, 170, 33], [250, 170, 34],
                 [128, 128, 128], [250, 170, 35], [102, 102, 156],
                 [128, 64, 255], [140, 140, 200], [170, 170, 170],
                 [250, 170, 36], [250, 170, 160], [250, 170, 37], [96, 96, 96],
                 [230, 150, 140], [128, 64, 128], [110, 110, 110],
                 [110, 110, 110], [244, 35, 232], [128, 196,
                                                   128], [150, 100, 100],
                 [70, 70, 70], [150, 150, 150], [150, 120, 90], [220, 20, 60],
                 [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200],
                 [255, 255, 255], [255, 255, 255], [250, 170, 29],
                 [250, 170, 28], [250, 170, 26], [250, 170,
                                                  25], [250, 170, 24],
                 [250, 170, 22], [250, 170, 21], [250, 170,
                                                  20], [255, 255, 255],
                 [250, 170, 19], [250, 170, 18], [250, 170,
                                                  12], [250, 170, 11],
                 [255, 255, 255], [255, 255, 255], [250, 170, 16],
                 [250, 170, 15], [250, 170, 15], [255, 255, 255],
                 [255, 255, 255], [255, 255, 255], [255, 255, 255],
                 [64, 170, 64], [230, 160, 50],
                 [70, 130, 180], [190, 255, 255], [152, 251, 152],
                 [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30],
                 [100, 140, 180], [220, 128, 128], [222, 40,
                                                    40], [100, 170, 30],
                 [40, 40, 40], [33, 33, 33], [100, 128, 160], [20, 20, 255],
                 [142, 0, 0], [70, 100, 150], [250, 171, 30], [250, 172, 30],
                 [250, 173, 30], [250, 174, 30], [250, 175,
                                                  30], [250, 176, 30],
                 [210, 170, 100], [153, 153, 153], [153, 153, 153],
                 [128, 128, 128], [0, 0, 80], [210, 60, 60], [250, 170, 30],
                 [250, 170, 30], [250, 170, 30], [250, 170,
                                                  30], [250, 170, 30],
                 [250, 170, 30], [192, 192, 192], [192, 192, 192],
                 [192, 192, 192], [220, 220, 0], [220, 220, 0], [0, 0, 196],
                 [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32],
                 [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90],
                 [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
                 [0, 0, 70], [0, 0, 142], [0, 0, 192], [170, 170, 170],
                 [32, 32, 32], [111, 74, 0], [120, 10, 10], [81, 0, 81],
                 [111, 111, 0], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
