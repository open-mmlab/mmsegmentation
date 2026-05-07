"""
Sen2GF3Floods dataset - fused Sentinel-2 + GF-3 flood segmentation.

Expected layout::

    data/Sen2GF3Floods/Sen2GF3Floods/
        sentinel2/<name>.tif    # 4-band (R, G, B, NIR)
        gaofen3/<name>.tif      # 2-band (HH, HV)
        label/<name>.tif        # 1-band (0=bg, 1=flood)
        splits/{train,val,test}.txt   # one filename per line

The loading transform ``LoadSen2GF3FloodsImage`` fuses sentinel2 and
gaofen3 into a single 6-band (H, W, 6) array. Normalization uses
the ``'sen2gf3'`` key in ``MultiModalNormalize.NORM_CONFIGS``.
"""
import os.path as osp
from typing import List

import mmengine
import mmengine.fileio as fileio
from mmengine.logging import print_log

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Sen2GF3FloodsDataset(BaseSegDataset):
    """Sen2GF3Floods dataset for fused optical+SAR fine-tuning.

    Args:
        ann_file (str, optional): Split file (one filename per line).
        **kwargs: forwarded to :class:`BaseSegDataset`.
    """

    METAINFO = dict(
        classes=('Background', 'Flood'),
        palette=[[0, 0, 0], [255, 0, 0]],
    )

    S2_SUBDIR = 'sentinel2'
    GF3_SUBDIR = 'gaofen3'
    LABEL_SUBDIR = 'label'
    CHANNELS = 6

    def __init__(self, **kwargs):
        kwargs.setdefault('img_suffix', '.tif')
        kwargs.setdefault('seg_map_suffix', '.tif')
        kwargs.setdefault('reduce_zero_label', False)

        data_prefix = kwargs.pop('data_prefix', None)
        if not data_prefix:
            data_prefix = dict(
                img_path=self.S2_SUBDIR,
                seg_map_path=self.LABEL_SUBDIR,
            )

        super().__init__(data_prefix=data_prefix, **kwargs)

    def load_data_list(self) -> List[dict]:
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)

        data_list = []
        ann_file = self.ann_file
        use_ann_file = bool(ann_file) and osp.isfile(ann_file)

        if use_ann_file:
            lines = mmengine.list_from_file(
                ann_file, backend_args=self.backend_args)
            for line in lines:
                name = line.strip()
                if not name:
                    continue
                data_list.append(self._build_info(name, ann_dir))
        else:
            for f in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix='.tif',
                    recursive=False,
                    backend_args=self.backend_args):
                data_list.append(self._build_info(f, ann_dir))

        data_list = sorted(data_list, key=lambda x: x['img_path'])

        print_log(
            f'[Sen2GF3Floods] loaded {len(data_list)} samples',
            logger='current')
        return data_list

    def _build_info(self, name: str, ann_dir: str) -> dict:
        data_root = self.data_root
        info = dict(
            img_path=osp.join(data_root, self.S2_SUBDIR, name),
            gf3_path=osp.join(data_root, self.GF3_SUBDIR, name),
            modal_type='sen2gf3',
            actual_channels=self.CHANNELS,
            dataset_source=0,
            dataset_name='sen2gf3',
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label,
            seg_fields=[],
        )
        if ann_dir is not None:
            info['seg_map_path'] = osp.join(data_root, self.LABEL_SUBDIR, name)
        return info
