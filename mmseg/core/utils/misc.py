# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn.functional as F

from mmseg.core import SegDataSample


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def stack_batch(inputs: List[torch.Tensor],
                batch_data_samples: List[SegDataSample],
                size: tuple = None,
                pad_value: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255,
                padding_mode: str = 'constant') -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        batch_data_samples (list[:obj:`SegDataSample`]): The Data
            Samples. It usually includes information such as `gt_sem_seg`.
        size (tuple): The img crop size.
        pad_value (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.

    Returns:
       Tensor: The 4D-tensor.
       batch_data_samples (list[:obj:`SegDataSample`]): After the padding of
            the gt_seg_map.
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len(set([tensor.ndim for tensor in inputs])) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len(set([tensor.shape[0] for tensor in inputs])) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    padded_samples = []

    for tensor, data_sample in zip(inputs, batch_data_samples):
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]
        if sum(padding_size) == 0:
            padded_samples.append(tensor)
        else:
            # pad img
            pad_img = F.pad(
                tensor, padding_size, mode=padding_mode, value=pad_value)
            padded_samples.append(pad_img)
            # pad gt_sem_seg
            gt_sem_seg = data_sample.gt_sem_seg.data
            gt_width = max(pad_img.shape[-1] - gt_sem_seg.shape[-1], 0)
            gt_height = max(pad_img.shape[-2] - gt_sem_seg.shape[-2], 0)
            padding_gt_size = (0, gt_width, 0, gt_height)
            del data_sample.gt_sem_seg.data
            data_sample.gt_sem_seg.data = F.pad(
                gt_sem_seg,
                padding_gt_size,
                mode=padding_mode,
                value=seg_pad_val)

    return torch.stack(padded_samples, dim=0), batch_data_samples
