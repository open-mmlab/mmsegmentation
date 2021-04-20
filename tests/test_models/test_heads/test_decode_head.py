from unittest.mock import patch

import pytest
import torch

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from .utils import to_cuda


@patch.multiple(BaseDecodeHead, __abstractmethods__=set())
def test_decode_head():

    with pytest.raises(AssertionError):
        # default input_transform doesn't accept multiple inputs
        BaseDecodeHead([32, 16], 16, num_classes=19)

    with pytest.raises(AssertionError):
        # default input_transform doesn't accept multiple inputs
        BaseDecodeHead(32, 16, num_classes=19, in_index=[-1, -2])

    with pytest.raises(AssertionError):
        # supported mode is resize_concat only
        BaseDecodeHead(32, 16, num_classes=19, input_transform='concat')

    with pytest.raises(AssertionError):
        # in_channels should be list|tuple
        BaseDecodeHead(32, 16, num_classes=19, input_transform='resize_concat')

    with pytest.raises(AssertionError):
        # in_index should be list|tuple
        BaseDecodeHead([32],
                       16,
                       in_index=-1,
                       num_classes=19,
                       input_transform='resize_concat')

    with pytest.raises(AssertionError):
        # len(in_index) should equal len(in_channels)
        BaseDecodeHead([32, 16],
                       16,
                       num_classes=19,
                       in_index=[-1],
                       input_transform='resize_concat')

    # test default dropout
    head = BaseDecodeHead(32, 16, num_classes=19)
    assert hasattr(head, 'dropout') and head.dropout.p == 0.1

    # test set dropout
    head = BaseDecodeHead(32, 16, num_classes=19, dropout_ratio=0.2)
    assert hasattr(head, 'dropout') and head.dropout.p == 0.2

    # test no input_transform
    inputs = [torch.randn(1, 32, 45, 45)]
    head = BaseDecodeHead(32, 16, num_classes=19)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.in_channels == 32
    assert head.input_transform is None
    transformed_inputs = head._transform_inputs(inputs)
    assert transformed_inputs.shape == (1, 32, 45, 45)

    # test input_transform = resize_concat
    inputs = [torch.randn(1, 32, 45, 45), torch.randn(1, 16, 21, 21)]
    head = BaseDecodeHead([32, 16],
                          16,
                          num_classes=19,
                          in_index=[0, 1],
                          input_transform='resize_concat')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.in_channels == 48
    assert head.input_transform == 'resize_concat'
    transformed_inputs = head._transform_inputs(inputs)
    assert transformed_inputs.shape == (1, 48, 45, 45)
