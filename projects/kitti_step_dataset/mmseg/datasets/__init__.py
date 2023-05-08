# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.ade import ADE20KDataset
from mmseg.datasets.builder import (DATASETS, PIPELINES, build_dataloader,
                                    build_dataset)
from mmseg.datasets.chase_db1 import ChaseDB1Dataset
from mmseg.datasets.cityscapes import CityscapesDataset
from mmseg.datasets.coco_stuff import COCOStuffDataset
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.dark_zurich import DarkZurichDataset
from mmseg.datasets.dataset_wrappers import (ConcatDataset,
                                             MultiImageMixDataset,
                                             RepeatDataset)
from mmseg.datasets.drive import DRIVEDataset
from mmseg.datasets.face import FaceOccludedDataset
from mmseg.datasets.hrf import HRFDataset
from mmseg.datasets.imagenets import (ImageNetSDataset,
                                      LoadImageNetSAnnotations,
                                      LoadImageNetSImageFromFile)
from mmseg.datasets.isaid import iSAIDDataset
from mmseg.datasets.isprs import ISPRSDataset
from mmseg.datasets.loveda import LoveDADataset
from mmseg.datasets.night_driving import NightDrivingDataset
from mmseg.datasets.pascal_context import (PascalContextDataset,
                                           PascalContextDataset59)
from mmseg.datasets.potsdam import PotsdamDataset
from mmseg.datasets.stare import STAREDataset
from mmseg.datasets.voc import PascalVOCDataset
from .kitti_step import KITTISTEPDataset

__all__ = [
    'CustomDataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'PascalVOCDataset',
    'ADE20KDataset',
    'PascalContextDataset',
    'PascalContextDataset59',
    'ChaseDB1Dataset',
    'DRIVEDataset',
    'HRFDataset',
    'STAREDataset',
    'DarkZurichDataset',
    'NightDrivingDataset',
    'COCOStuffDataset',
    'LoveDADataset',
    'MultiImageMixDataset',
    'iSAIDDataset',
    'ISPRSDataset',
    'PotsdamDataset',
    'FaceOccludedDataset',
    'ImageNetSDataset',
    'LoadImageNetSAnnotations',
    'LoadImageNetSImageFromFile',
    'KITTISTEPDataset',
]
