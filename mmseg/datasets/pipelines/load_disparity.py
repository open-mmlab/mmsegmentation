# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadCityscapesDisparity(BaseTransform):
    """Load Cityscapes disparity maps and normalize them.

    This transform infers the disparity file path from the paired RGB image
    path following the official Cityscapes naming convention. The disparity is
    expected to be stored as a 16-bit PNG file. After loading, a logarithmic
    normalization is applied so that the values fall into the range [0, 1].

    Required Keys:
        - img_path or filename

    Added Keys:
        - disp (np.ndarray): Normalized disparity map with shape (H, W).
        - disp_path (str): Absolute disparity file path.
        - disp_shape (Tuple[int, int]): Height and width of the disparity map.
        - disp_dtype (np.dtype): Original data type before normalization.

    Args:
        max_disparity (Optional[float]): Optional prior upper bound for the
            disparity values. When provided, the normalization uses this
            constant instead of the per-image maximum for stability across the
            dataset. Defaults to ``None``.
        log_eps (float): Small epsilon added before taking logarithm to avoid
            numerical issues. Defaults to 1e-3.
        backend_args (Optional[dict]): Arguments to instantiate a file backend
            when reading disparity files. Defaults to ``None``.
        imdecode_backend (str): Backend used by :func:`mmcv.imfrombytes` to
            decode PNG files. Defaults to ``'pillow'``.
    """

    def __init__(
        self,
        max_disparity: Optional[float] = None,
        log_eps: float = 1e-3,
        backend_args: Optional[dict] = None,
        imdecode_backend: str = 'pillow',
    ) -> None:
        super().__init__()
        self.max_disparity = max_disparity
        self.log_eps = log_eps
        self.backend_args = backend_args
        self.imdecode_backend = imdecode_backend

    @staticmethod
    def infer_disp_path(img_path: str) -> str:
        """Infer the Cityscapes disparity path from the RGB image path."""
        if 'leftImg8bit' not in img_path:
            raise ValueError('The provided img_path does not follow the '
                             'Cityscapes leftImg8bit naming convention: '
                             f"{img_path}")
        disp_path = img_path.replace('leftImg8bit', 'disparity')
        disp_path = disp_path.replace('_leftImg8bit', '_disparity')
        return disp_path

    def transform(self, results: dict) -> dict:
        img_path = results.get('img_path') or results.get('filename')
        if img_path is None:
            raise KeyError('`img_path` or `filename` is required to locate the '
                           'paired Cityscapes disparity map.')
        if not osp.isabs(img_path) and results.get('data_root'):
            img_path = osp.join(results['data_root'], img_path)

        disp_path = self.infer_disp_path(img_path)
        disp_bytes = fileio.get(disp_path, backend_args=self.backend_args)
        disparity = mmcv.imfrombytes(
            disp_bytes, flag='unchanged', backend=self.imdecode_backend)
        if disparity.ndim == 3:
            disparity = disparity[..., 0]
        original_dtype = disparity.dtype
        disparity = disparity.astype(np.float32)

        valid_mask = disparity > 0
        disp_norm = np.zeros_like(disparity, dtype=np.float32)
        if np.any(valid_mask):
            disparity_valid = disparity[valid_mask]
            disp_log = np.log(disparity_valid + self.log_eps)
            if self.max_disparity is not None:
                denom = np.log(self.max_disparity + self.log_eps)
            else:
                denom = disp_log.max()
            denom = max(denom, np.finfo(np.float32).eps)
            disp_norm[valid_mask] = disp_log / denom
        disp_norm = np.clip(disp_norm, 0.0, 1.0)

        results['disp'] = disp_norm
        results['disp_path'] = disp_path
        results['disp_shape'] = disp_norm.shape
        results['disp_dtype'] = original_dtype
        disp_fields = results.setdefault('disp_fields', [])
        if 'disp' not in disp_fields:
            disp_fields.append('disp')
        return results

    def __repr__(self) -> str:
        repr_str = (f"{self.__class__.__name__}(max_disparity={self.max_disparity}, "
                    f"log_eps={self.log_eps}, "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f"backend_args={self.backend_args})")
        return repr_str
