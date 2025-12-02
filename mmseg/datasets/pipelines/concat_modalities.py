# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ConcatRGBDispTo4Ch(BaseTransform):
    """Concatenate RGB images and disparity maps to form 4-channel inputs.

    Required Keys:
        - img (np.ndarray): RGB image with shape (H, W, 3).
        - disp (np.ndarray): Normalized disparity map with shape (H, W).

    Modified Keys:
        - img (np.ndarray): Concatenated 4-channel tensor with shape (H, W, 4).

    Args:
        keep_rgb_dtype (bool): Whether to keep the RGB image dtype after
            concatenation. If ``False``, the output will be converted to
            ``np.float32``. Defaults to ``True``.
        disp_scale (Optional[float]): Optional scaling factor applied to the
            disparity channel after concatenation. Defaults to ``None`` (no
            scaling).
        delete_disp (bool): Whether to remove the standalone disparity entry
            from the results dict after concatenation. Defaults to ``True``.
    """

    def __init__(
        self,
        keep_rgb_dtype: bool = True,
        disp_scale: Optional[float] = None,
        delete_disp: bool = True,
    ) -> None:
        super().__init__()
        self.keep_rgb_dtype = keep_rgb_dtype
        self.disp_scale = disp_scale
        self.delete_disp = delete_disp

    def transform(self, results: dict) -> dict:
        if 'img' not in results:
            raise KeyError('`img` is required before applying '
                           '`ConcatRGBDispTo4Ch`.')
        if 'disp' not in results:
            raise KeyError('`disp` is required before applying '
                           '`ConcatRGBDispTo4Ch`. Please use '
                           '`LoadCityscapesDisparity` first.')

        img = results['img']
        disp = results['disp']
        if disp.ndim == 2:
            disp = disp[..., None]
        elif disp.ndim == 3 and disp.shape[-1] != 1:
            raise ValueError('Disparity map must be single channel. '
                             f'Got shape {disp.shape}.')

        if not self.keep_rgb_dtype or img.dtype != np.float32:
            img = img.astype(np.float32)
        disp = disp.astype(img.dtype)
        if self.disp_scale is not None:
            disp = disp * self.disp_scale

        img_4ch = np.concatenate([img, disp], axis=2)
        results['img'] = img_4ch
        h, w, _ = img_4ch.shape
        results['img_shape'] = (h, w, 4)
        if 'ori_shape' in results:
            ori = results['ori_shape']
            if isinstance(ori, tuple) and len(ori) >= 2:
                results['ori_shape'] = (ori[0], ori[1], 4)
            else:
                results['ori_shape'] = (h, w, 4)
        results['num_channels'] = 4
        if self.delete_disp:
            results.pop('disp', None)
            disp_fields = results.get('disp_fields', None)
            if isinstance(disp_fields, list) and 'disp' in disp_fields:
                disp_fields.remove('disp')
        return results

    def __repr__(self) -> str:
        repr_str = (f"{self.__class__.__name__}(keep_rgb_dtype={self.keep_rgb_dtype}, "
                    f"disp_scale={self.disp_scale}, "
                    f"delete_disp={self.delete_disp})")
        return repr_str
