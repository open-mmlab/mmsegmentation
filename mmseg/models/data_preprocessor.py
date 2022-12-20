# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            assert len(inputs) == 1, (
                'Batch inference is not support currently, '
                'as the image size might be different in a batch')
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)


@MODELS.register_module()
class MultiBranchDataPreprocessor(BaseDataPreprocessor):
    """DataPreprocessor wrapper for multi-branch data.

    Take semi-supervised object detection as an example, assume that
    the ratio of labeled data and unlabeled data in a batch is 1:1,
    `sup` indicates the branch where the labeled data is augmented,
    `unsup` indicate the branches where
    the unlabeled data is augmented by different pipeline.

    The input format of multi-branch data is shown as below :

    .. code-block:: none
        {
            'inputs':
                {
                    'sup': [Tensor, None],
                    'unsup': [None, Tensor],
                },
            'data_sample':
                {
                    'sup': [DetDataSample, None],
                    'unsup': [None, DetDataSample],
                }
        }

    The format of multi-branch data
    after filtering None is shown as below :

    .. code-block:: none
        {
            'inputs':
                {
                    'sup': [Tensor],
                    'unsup': [Tensor],
                },
            'data_sample':
                {
                    'sup': [DetDataSample],
                    'unsup': [DetDataSample],
                }
        }

    In order to reuse `SegDataPreprocessor` for the data
    from different branches, the format of multi-branch data
    grouped by branch is as below :

    .. code-block:: none
        {
            'sup':
                {
                    'inputs': [Tensor]
                    'data_sample': [DetDataSample]
                },
            'unsup':
                {
                    'inputs': [Tensor]
                    'data_sample': [DetDataSample]
                },
        }

    Args:
        data_preprocessor (:obj:`ConfigDict` or dict): Config of
            :class:`DetDataPreprocessor` to process the input data.
    """

    def __init__(self, data_preprocessor: ConfigType) -> None:
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor`` for multi-branch data.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict:

            - 'inputs' (Dict[str, obj:`torch.Tensor`]): The forward data of
                models from different branches.
            - 'data_sample' (Dict[str, obj:`DetDataSample`]): The annotation
                info of the sample from different branches.
        """

        if training is False:
            return self.data_preprocessor(data, training)

        # Filter out branches with a value of None
        for key in data.keys():
            for branch in data[key].keys():
                data[key][branch] = list(
                    filter(lambda x: x is not None, data[key][branch]))

        # Group data by branch
        multi_branch_data = {}
        for key in data.keys():
            for branch in data[key].keys():
                if multi_branch_data.get(branch, None) is None:
                    multi_branch_data[branch] = {key: data[key][branch]}
                elif multi_branch_data[branch].get(key, None) is None:
                    multi_branch_data[branch][key] = data[key][branch]
                else:
                    multi_branch_data[branch][key].append(data[key][branch])

        # Preprocess data from different branches
        for branch, _data in multi_branch_data.items():
            multi_branch_data[branch] = self.data_preprocessor(_data, training)

        # Format data by inputs and data_samples
        format_data = {}
        for branch in multi_branch_data.keys():
            for key in multi_branch_data[branch].keys():
                if format_data.get(key, None) is None:
                    format_data[key] = {branch: multi_branch_data[branch][key]}
                elif format_data[key].get(branch, None) is None:
                    format_data[key][branch] = multi_branch_data[branch][key]
                else:
                    format_data[key][branch].append(
                        multi_branch_data[branch][key])

        return format_data

    @property
    def device(self):
        return self.data_preprocessor.device

    def to(self, device: Optional[Union[int, torch.device]], *args,
           **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Args:
            device (int or torch.device, optional): The desired device of the
                parameters and buffers in this module.

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.to(device, *args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.cpu(*args, **kwargs)
