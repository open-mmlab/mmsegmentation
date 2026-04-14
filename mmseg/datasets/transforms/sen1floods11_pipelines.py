"""
Sen1Floods11 loading transforms.

The ``LabelHand`` TIFFs store signed values in ``{-1, 0, 1}``:

    -1: nodata  -> mapped to ``ignore_index`` (255 by default)
     0: non-flood / background
     1: flood

Default :class:`LoadAnnotations` uses ``mmcv.imfrombytes`` which is not
reliable for signed-int TIFF, so we decode with :mod:`tifffile` and
explicitly remap nodata. The returned ``gt_seg_map`` is ``np.uint8``
which is what the rest of the mmseg 1.x pipeline expects.
"""
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS

try:
    import tifffile
except ImportError:
    tifffile = None


@TRANSFORMS.register_module()
class LoadSen1Floods11Annotation(BaseTransform):
    """Load a Sen1Floods11 LabelHand TIFF segmentation map.

    Required Keys:
        - seg_map_path

    Added Keys:
        - gt_seg_map (np.uint8, shape (H, W))
        - seg_fields (list)

    Args:
        ignore_index (int): Value that replaces ``nodata_value`` in the
            output mask. Default: 255.
        nodata_value (int): Raw label value that indicates "no data".
            Default: -1.
    """

    def __init__(self, ignore_index: int = 255, nodata_value: int = -1):
        if tifffile is None:
            raise ImportError(
                'tifffile is required to load Sen1Floods11 labels. '
                'Install with `pip install tifffile`.')
        self.ignore_index = int(ignore_index)
        self.nodata_value = int(nodata_value)

    def transform(self, results: dict) -> dict:
        seg_path = results['seg_map_path']
        raw = tifffile.imread(seg_path)

        # Some TIFFs come as (1, H, W) / (H, W, 1)
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise RuntimeError(
                f'Expected a 2D label for {seg_path}, got shape {raw.shape}')

        raw = raw.astype(np.int32)

        gt = np.full(raw.shape, self.ignore_index, dtype=np.uint8)
        gt[raw == 0] = 0
        gt[raw == 1] = 1
        # Explicit nodata remap - catches anything that isn't {0, 1}.
        gt[raw == self.nodata_value] = self.ignore_index

        # Optional label remapping (preserves parity with LoadAnnotations)
        if results.get('label_map', None):
            gt_copy = gt.copy()
            for old_id, new_id in results['label_map'].items():
                gt[gt_copy == old_id] = new_id

        results['gt_seg_map'] = gt
        results.setdefault('seg_fields', [])
        if 'gt_seg_map' not in results['seg_fields']:
            results['seg_fields'].append('gt_seg_map')

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'ignore_index={self.ignore_index}, '
                f'nodata_value={self.nodata_value})')
