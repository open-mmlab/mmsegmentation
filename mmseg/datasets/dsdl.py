# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Sequence, Union

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

try:
    from dsdl.dataset import DSDLDataset
except ImportError:
    DSDLDataset = None


@DATASETS.register_module()
class DSDLSegDataset(BaseSegDataset):
    """Dataset for dsdl segmentation.

    Args:
        specific_key_path(dict): Path of specific key which can not
            be loaded by it's field name.
        pre_transform(dict): pre-transform functions before loading.
        used_labels(sequence): list of actual used classes in train steps,
            this must be subset of class domain.
    """

    METAINFO = {}

    def __init__(self,
                 specific_key_path: Dict = {},
                 pre_transform: Dict = {},
                 used_labels: Optional[Sequence] = None,
                 **kwargs) -> None:

        if DSDLDataset is None:
            raise RuntimeError(
                'Package dsdl is not installed. Please run "pip install dsdl".'
            )
        self.used_labels = used_labels

        loc_config = dict(type='LocalFileReader', working_dir='')
        if kwargs.get('data_root'):
            kwargs['ann_file'] = os.path.join(kwargs['data_root'],
                                              kwargs['ann_file'])
        required_fields = ['Image', 'LabelMap']

        self.dsdldataset = DSDLDataset(
            dsdl_yaml=kwargs['ann_file'],
            location_config=loc_config,
            required_fields=required_fields,
            specific_key_path=specific_key_path,
            transform=pre_transform,
        )
        BaseSegDataset.__init__(self, **kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load data info from a dsdl yaml file named as ``self.ann_file``

        Returns:
            List[dict]: A list of data list.
        """

        if self.used_labels:
            self._metainfo['classes'] = tuple(self.used_labels)
            self.label_map = self.get_label_map(self.used_labels)
        else:
            self._metainfo['classes'] = tuple(['background'] +
                                              self.dsdldataset.class_names)
        data_list = []

        for i, data in enumerate(self.dsdldataset):
            datainfo = dict(
                img_path=os.path.join(self.data_prefix['img_path'],
                                      data['Image'][0].location),
                seg_map_path=os.path.join(self.data_prefix['seg_map_path'],
                                          data['LabelMap'][0].location),
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            data_list.append(datainfo)

        return data_list

    def get_label_map(self,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        """Require label mapping.

        The ``label_map`` is a dictionary, its keys are the old label ids and
        its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in class_dom
        is not equal to new classes in args and nether of them is not
        None, `label_map` is not None.
        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.
        Returns:
            dict, optional: The mapping from old classes to new classes.
        """
        old_classes = ['background'] + self.dsdldataset.class_names
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(old_classes):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in class_dom.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None
