from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class NYUv2Dataset(CustomDataset):
    """ The  NYU-v2 dataset is a popular RGB-D dataset. 
    The dataset contains 1449 RGB-D images in total. 
    By convention, split the dataset into  795 training images and 654 testing images.
    Also adopt the 40-class setting to re-label the image pixels.
    """
    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
               'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
               'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
               'television', 'paper', 'towel', 'shower curtain', 'box',
               'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp',
               'bathtub', 'bag', 'otherstructure', 'otherfurniture',
               'otherprop')

    def __init__(self, **kwargs):
        super(NYUv2Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
