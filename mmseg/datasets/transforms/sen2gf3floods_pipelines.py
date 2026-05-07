"""
Sen2GF3Floods loading transforms.

``LoadSen2GF3FloodsImage`` fuses sentinel2 (4-band RGBN) and gaofen3
(2-band HH/HV) TIFFs into a single 6-channel (H, W, 6) float32 array.

``LoadSen2GF3FloodsAnnotation`` loads the label TIFF (0=bg, 1=flood).
"""
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS

try:
    import tifffile
except ImportError:
    tifffile = None


@TRANSFORMS.register_module()
class LoadSen2GF3FloodsImage(BaseTransform):
    """Load and fuse Sentinel-2 + GF-3 into a 6-band image.

    Required Keys:
        - img_path (str): path to sentinel2 TIFF (4 bands)
        - gf3_path (str): path to gaofen3 TIFF (2 bands)

    Added Keys:
        - img (np.ndarray, H×W×6, float32)
        - img_shape, ori_shape, ori_filename
    """

    def __init__(self, to_float32: bool = True):
        if tifffile is None:
            raise ImportError('tifffile is required')
        self.to_float32 = to_float32

    @staticmethod
    def _load_hwc(path: str) -> np.ndarray:
        img = tifffile.imread(path).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.ndim == 3 and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))
        return img

    def transform(self, results: dict) -> dict:
        import os.path as osp

        s2 = self._load_hwc(results['img_path'])
        gf3 = self._load_hwc(results['gf3_path'])

        fused = np.concatenate([s2, gf3], axis=-1)  # (H, W, 6)

        if self.to_float32:
            fused = fused.astype(np.float32)

        results['img'] = fused
        results['img_shape'] = fused.shape[:2]
        results['ori_shape'] = fused.shape[:2]
        results['ori_filename'] = osp.basename(results['img_path'])
        results['actual_channels'] = fused.shape[2]

        return results


@TRANSFORMS.register_module()
class LoadSen2GF3FloodsAnnotation(BaseTransform):
    """Load a Sen2GF3Floods label TIFF (0=bg, 1=flood).

    Required Keys:
        - seg_map_path

    Added Keys:
        - gt_seg_map (np.uint8, H×W)
        - seg_fields (list)

    Args:
        ignore_index (int): Value for pixels outside {0, 1}.
            Default: 255.
    """

    def __init__(self, ignore_index: int = 255):
        if tifffile is None:
            raise ImportError('tifffile is required')
        self.ignore_index = int(ignore_index)

    def transform(self, results: dict) -> dict:
        raw = tifffile.imread(results['seg_map_path'])
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise RuntimeError(
                f'Expected 2D label, got shape {raw.shape} '
                f'from {results["seg_map_path"]}')

        raw = raw.astype(np.int32)
        gt = np.full(raw.shape, self.ignore_index, dtype=np.uint8)
        gt[raw == 0] = 0
        gt[raw == 1] = 1

        if results.get('label_map', None):
            gt_copy = gt.copy()
            for old_id, new_id in results['label_map'].items():
                gt[gt_copy == old_id] = new_id

        results['gt_seg_map'] = gt
        results.setdefault('seg_fields', [])
        if 'gt_seg_map' not in results['seg_fields']:
            results['seg_fields'].append('gt_seg_map')

        return results
