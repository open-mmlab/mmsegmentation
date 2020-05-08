import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module
class PascalContextDataset(CustomDataset):
    CLASSES = ('background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
               'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
               'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
               'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
               'floor', 'flower', 'food', 'grass', 'ground', 'horse',
               'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
               'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
               'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
               'diningtable', 'track', 'train', 'tree', 'truck', 'tvmonitor',
               'wall', 'water', 'window', 'wood')

    def __init__(self, split, **kwargs):
        super(PascalContextDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
