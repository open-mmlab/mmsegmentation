import pytest
import torch

from mmseg.models.decode_heads import DPTHead


def test_dpt_head():

    with pytest.raises(AssertionError):
        # input_transform must be 'multiple_select'
        head = DPTHead(
            in_channels=[768, 768, 768, 768],
            channels=4,
            num_classes=19,
            in_index=[0, 1, 2, 3])

    head = DPTHead(
        in_channels=[768, 768, 768, 768],
        channels=4,
        num_classes=19,
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select')

    inputs = [[torch.randn(4, 768, 2, 2),
               torch.randn(4, 768)] for _ in range(4)]
    output = head(inputs)
    assert output.shape == torch.Size((4, 19, 16, 16))

    # test readout operation
    head = DPTHead(
        in_channels=[768, 768, 768, 768],
        channels=4,
        num_classes=19,
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        readout_type='add')
    output = head(inputs)
    assert output.shape == torch.Size((4, 19, 16, 16))

    head = DPTHead(
        in_channels=[768, 768, 768, 768],
        channels=4,
        num_classes=19,
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        readout_type='project')
    output = head(inputs)
    assert output.shape == torch.Size((4, 19, 16, 16))
