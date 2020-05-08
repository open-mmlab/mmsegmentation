import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(to_float32={self.to_float32}, ' \
               f"color_type='{self.color_type}')"


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles(object):
    """ Load multi channel images from a list of separate channel files.
    Expects results['filename'] to be a list of filenames
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(to_float32={self.to_float32}, ' \
               f"color_type='{self.color_type}')"


@PIPELINES.register_module()
class LoadAnnotations(object):

    def __init__(self, with_seg=True, use_pil=False, reduce_zero_label=False):
        self.with_seg = with_seg
        self.use_pil = use_pil
        self.reduce_zero_label = reduce_zero_label

    def _load_semantic_seg(self, results):
        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        if self.use_pil:
            gt_semantic_seg = np.array(Image.open(filename), dtype=np.uint8)
        else:
            gt_semantic_seg = mmcv.imread(
                filename, flag='unchanged').squeeze().astype(np.uint8)
        # reduce zero_label
        if self.reduce_zero_label:
            gt_semantic_seg = gt_semantic_seg - 1
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_seg={self.with_seg})'
        return repr_str
