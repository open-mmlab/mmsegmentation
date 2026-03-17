"""
Multi-Modal Deepflood Dataset - MMSeg 1.x Version

Key changes from 0.x:
- Base class: CustomDataset -> BaseSegDataset
- Registry: from builder import DATASETS -> from mmseg.registry import DATASETS
- load_annotations() -> load_data_list()
- img_dir/ann_dir -> data_prefix dict
- img_infos -> data_list
- evaluate() -> IoUMetric (external evaluator)
"""
import os.path as osp
from collections import OrderedDict
from typing import List

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.logging import print_log

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MultiModalDeepflood(BaseSegDataset):
    """Multi-Modal Deepflood dataset for flood segmentation.

    Supports SAR, RGB, and GaoFen (GF) modalities with automatic
    modality identification from filenames.
    """

    METAINFO = dict(
        classes=('Background', 'Flood'),
        palette=[[0, 0, 0], [255, 0, 0]]
    )

    # Modal configuration
    MODAL_CONFIGS = {
        'sar': {'channels': 8, 'pattern': 'sar'},
        'rgb': {'channels': 3, 'pattern': 'rgb'},
        'GF': {'channels': 5, 'pattern': 'GF'},
    }

    MODAL_TO_DATASET = {
        'sar': {'dataset_source': 0, 'dataset_name': 'sar'},
        'rgb': {'dataset_source': 1, 'dataset_name': 'rgb'},
        'GF': {'dataset_source': 2, 'dataset_name': 'GF'},
    }

    def __init__(self, **kwargs):
        # Set default suffixes for flood data
        kwargs.setdefault('img_suffix', '.tif')
        kwargs.setdefault('seg_map_suffix', '.png')
        kwargs.setdefault('reduce_zero_label', False)
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory.

        Returns:
            list[dict]: All data info of dataset, each dict contains:
                - img_path (str)
                - seg_map_path (str)
                - modal_type (str)
                - actual_channels (int)
                - dataset_source (int)
                - dataset_name (str)
                - label_map (dict or None)
                - reduce_zero_label (bool)
                - seg_fields (list)
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)

        if not osp.isdir(self.ann_file) and self.ann_file:
            # Load from annotation file
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir,
                                      img_name + self.img_suffix))

                # Identify modality
                modal_info = self._identify_modality(img_name)
                data_info.update(modal_info)

                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)

                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            # Scan directory
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))

                # Identify modality
                modal_info = self._identify_modality(img)
                data_info.update(modal_info)

                if ann_dir is not None:
                    seg_map = img[:-len(self.img_suffix)] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)

                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)

            data_list = sorted(data_list, key=lambda x: x['img_path'])

        print_log(f'Loaded {len(data_list)} images', logger='current')
        self._print_modal_statistics(data_list)

        return data_list

    def _identify_modality(self, img_name):
        """Identify modality from filename."""
        img_name_lower = img_name.lower()

        for modal_name, config in self.MODAL_CONFIGS.items():
            if config['pattern'].lower() in img_name_lower:
                dataset_info = self.MODAL_TO_DATASET.get(modal_name, {
                    'dataset_source': 1,
                    'dataset_name': 'rgb'
                })

                return {
                    'modal_type': modal_name,
                    'actual_channels': config['channels'],
                    'dataset_source': dataset_info['dataset_source'],
                    'dataset_name': dataset_info['dataset_name'],
                }

        # Default: RGB
        return {
            'modal_type': 'rgb',
            'actual_channels': 3,
            'dataset_source': 1,
            'dataset_name': 'rgb',
        }

    def _print_modal_statistics(self, data_list):
        """Print dataset modal statistics."""
        modal_counts = {}
        for info in data_list:
            modal = info['modal_type']
            modal_counts[modal] = modal_counts.get(modal, 0) + 1

        print_log("\n=== Dataset Modal Statistics ===", logger='current')
        for modal, count in sorted(modal_counts.items()):
            channels = self.MODAL_CONFIGS.get(
                modal, {}).get('channels', 'unknown')
            print_log(
                f"  {modal}: {count} images ({channels} channels)",
                logger='current')
        print_log("================================\n", logger='current')
