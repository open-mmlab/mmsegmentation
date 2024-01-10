# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

classes_exp = ('unlabelled', 'road', 'road marks', 'vegetation',
               'painted metal', 'sky', 'concrete', 'pedestrian', 'water',
               'unpainted metal', 'glass')
palette_exp = [[0, 0, 0], [77, 77, 77], [255, 255, 255], [0, 255, 0],
               [255, 0, 0], [0, 0, 255], [102, 51, 0], [255, 255, 0],
               [0, 207, 250], [255, 166, 0], [0, 204, 204]]


@DATASETS.register_module()
class HSIDrive20Dataset(BaseSegDataset):
    """HSI-Drive v2.0 (https://ieeexplore.ieee.org/document/10371793), the
    updated version of HSI-Drive
    (https://ieeexplore.ieee.org/document/9575298), is a structured dataset for
    the research and development of automated driving systems (ADS) supported
    by hyperspectral imaging (HSI). It contains per-pixel manually annotated
    images selected from videos recorded in real driving conditions and has
    been organized according to four parameters: season, daytime, road type,
    and weather conditions.

    The video sequences have been captured with a small-size 25-band VNIR
    (Visible-NearlnfraRed) snapshot hyperspectral camera mounted on a driving
    automobile. As a consequence, you need to modify the in_channels parameter
    of your model from 3 (RGB images) to 25 (HSI images) as it is done in
    configs/unet/unet-s5-d16_fcn_4xb4-160k_hsidrive-192x384.py

    Apart from the abovementioned articles, additional information is provided
    in the website (https://ipaccess.ehu.eus/HSI-Drive/) from where you can
    download the dataset and also visualize some examples of segmented videos.
    """

    METAINFO = dict(classes=classes_exp, palette=palette_exp)

    def __init__(self,
                 img_suffix='.npy',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
