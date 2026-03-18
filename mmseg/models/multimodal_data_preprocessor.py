"""
Multi-Modal Data PreProcessor - handles images with different channel counts.

Unlike SegDataPreProcessor which stacks all images into a [B,C,H,W] tensor,
this preprocessor keeps them as a list of [C_i,H,W] tensors since different
modalities have different channel counts (e.g., SAR:8ch, RGB:3ch, GF:5ch).
"""
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS


@MODELS.register_module()
class MultiModalDataPreProcessor(BaseDataPreprocessor):
    """Data pre-processor for multi-modal segmentation.

    Unlike SegDataPreProcessor, this does NOT stack inputs into a single
    tensor because different modalities may have different channel counts.
    Instead, it returns inputs as a list of tensors.

    Args:
        size (tuple, optional): Fixed padding size (H, W).
        pad_val (float): Padding value for images. Default: 0.
        seg_pad_val (float): Padding value for segmentation maps. Default: 255.
    """

    def __init__(
        self,
        size: Optional[tuple] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
    ):
        super().__init__()
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        data = self.cast_data(data)
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        inputs = [_input.float() for _input in inputs]

        # Pad spatial dimensions only (no stacking across channels)
        padded_inputs = []
        for i, tensor in enumerate(inputs):
            if self.size is not None:
                width = max(self.size[-1] - tensor.shape[-1], 0)
                height = max(self.size[-2] - tensor.shape[-2], 0)
                padding_size = (0, width, 0, height)
            else:
                padding_size = (0, 0, 0, 0)

            pad_img = F.pad(tensor, padding_size, value=self.pad_val)
            padded_inputs.append(pad_img)

            if data_samples is not None:
                ds = data_samples[i]
                if 'gt_sem_seg' in ds:
                    gt = ds.gt_sem_seg.data
                    del ds.gt_sem_seg.data
                    ds.gt_sem_seg.data = F.pad(
                        gt, padding_size, value=self.seg_pad_val)
                if hasattr(ds, 'gt_boundary_map') and 'gt_boundary_map' in ds:
                    gt_b = ds.gt_boundary_map.data
                    del ds.gt_boundary_map.data
                    ds.gt_boundary_map.data = F.pad(
                        gt_b, padding_size, value=self.seg_pad_val)

                ds.set_metainfo({
                    'img_shape': tensor.shape[-2:],
                    'pad_shape': pad_img.shape[-2:],
                    'padding_size': padding_size
                })

        # Return list (NOT stacked tensor) — the model handles the list
        return dict(inputs=padded_inputs, data_samples=data_samples)
