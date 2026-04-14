"""
Sen1Floods11 single-modal flood segmentation dataset - MMSeg 1.x.

Expected on-disk layout (the default from the official Sen1Floods11 release)::

    data/Sen1Floods11/
        S1Hand/
            <region>_<id>_S1Hand.tif      # 2-band SAR (VV, VH in dB)
        S2Hand/
            <region>_<id>_S2Hand.tif      # 13-band Sentinel-2 MSI
        LabelHand/
            <region>_<id>_LabelHand.tif   # 1-band label (-1=nodata, 0=bg, 1=flood)

All images are 512x512. Labels are signed TIFFs; the ``-1`` nodata value
is mapped to ``ignore_index=255`` by the companion
``LoadSen1Floods11Annotation`` transform.

This dataset plugs into the existing multi-modal Swin+MoE pipeline by
exposing ``modal_type`` / ``actual_channels`` / ``dataset_name`` on each
sample, so the only things a config needs to do to fine-tune on
Sen1Floods11 is to:

    1. set ``dataset_type='Sen1Floods11Dataset'`` and pick
       ``modality='s1'`` or ``'s2'``;
    2. register a new modal in ``model.backbone.modal_configs`` /
       ``training_modals`` with the matching channel count;
    3. use ``model.dataset_names=['s1']`` (or ``['s2']``) so the
       per-dataset decode head key matches.
"""
import os.path as osp
from typing import List

import mmengine
import mmengine.fileio as fileio
from mmengine.logging import print_log

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Sen1Floods11Dataset(BaseSegDataset):
    """Sen1Floods11 dataset for single-modal fine-tuning.

    Args:
        modality (str): ``'s1'`` (2-band SAR) or ``'s2'`` (13-band MSI).
        ann_file (str, optional): Path to a split file listing one
            sample base-name per line (e.g. ``Bolivia_23014``). The
            path may be relative to ``data_root``. If empty or absent,
            the whole ``data_prefix['img_path']`` directory is scanned.
        **kwargs: forwarded to :class:`BaseSegDataset`. ``img_suffix``,
            ``seg_map_suffix``, ``data_prefix`` all default to values
            that match the Sen1Floods11 layout described above.
    """

    METAINFO = dict(
        classes=('Background', 'Flood'),
        palette=[[0, 0, 0], [255, 0, 0]],
    )

    # Per-modality layout / shape info.
    MODAL_CONFIG = {
        's1': {
            'channels': 2,
            'img_subdir': 'S1Hand',
            'img_suffix': '_S1Hand.tif',
            'dataset_source': 0,
            'dataset_name': 's1',
        },
        's2': {
            'channels': 13,
            'img_subdir': 'S2Hand',
            'img_suffix': '_S2Hand.tif',
            'dataset_source': 1,
            'dataset_name': 's2',
        },
    }

    SEG_SUBDIR = 'LabelHand'
    SEG_SUFFIX = '_LabelHand.tif'

    def __init__(self, modality: str = 's1', **kwargs):
        if modality not in self.MODAL_CONFIG:
            raise ValueError(
                f'Unknown modality "{modality}". Expected one of '
                f'{list(self.MODAL_CONFIG.keys())}')
        self.modality = modality

        modal_cfg = self.MODAL_CONFIG[modality]

        kwargs.setdefault('img_suffix', modal_cfg['img_suffix'])
        kwargs.setdefault('seg_map_suffix', self.SEG_SUFFIX)
        kwargs.setdefault('reduce_zero_label', False)

        data_prefix = kwargs.pop('data_prefix', None)
        if not data_prefix:
            data_prefix = dict(
                img_path=modal_cfg['img_subdir'],
                seg_map_path=self.SEG_SUBDIR,
            )

        super().__init__(data_prefix=data_prefix, **kwargs)

    # ------------------------------------------------------------------
    # core list building
    # ------------------------------------------------------------------
    def load_data_list(self) -> List[dict]:
        modal_cfg = self.MODAL_CONFIG[self.modality]
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        assert img_dir is not None, \
            'Sen1Floods11Dataset requires data_prefix["img_path"]'

        data_list = []

        ann_file = self.ann_file
        use_ann_file = bool(ann_file) and osp.isfile(ann_file)

        if use_ann_file:
            lines = mmengine.list_from_file(
                ann_file, backend_args=self.backend_args)
            for line in lines:
                base = line.strip()
                if not base:
                    continue
                # Accept entries that include any of the known suffixes.
                for suf in (modal_cfg['img_suffix'], self.SEG_SUFFIX):
                    if base.endswith(suf):
                        base = base[:-len(suf)]
                        break
                img_name = base + modal_cfg['img_suffix']
                data_list.append(
                    self._build_info(img_name, img_dir, ann_dir))
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=modal_cfg['img_suffix'],
                    recursive=True,
                    backend_args=self.backend_args):
                data_list.append(
                    self._build_info(img, img_dir, ann_dir))

        data_list = sorted(data_list, key=lambda x: x['img_path'])

        print_log(
            f'[Sen1Floods11] modality="{self.modality}" '
            f'loaded {len(data_list)} images from "{img_dir}"',
            logger='current')

        return data_list

    def _build_info(self, img_name: str, img_dir: str,
                    ann_dir: str) -> dict:
        modal_cfg = self.MODAL_CONFIG[self.modality]
        info = dict(
            img_path=osp.join(img_dir, img_name),
            modal_type=self.modality,
            actual_channels=modal_cfg['channels'],
            dataset_source=modal_cfg['dataset_source'],
            dataset_name=modal_cfg['dataset_name'],
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label,
            seg_fields=[],
        )
        if ann_dir is not None:
            base = img_name[:-len(modal_cfg['img_suffix'])]
            info['seg_map_path'] = osp.join(ann_dir, base + self.SEG_SUFFIX)
        return info
