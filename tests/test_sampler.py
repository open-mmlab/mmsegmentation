import pytest
import torch

from mmseg.core import OHEMPixelSampler


def test_ohem_sampler():

    with pytest.raises(AssertionError):
        # seg_logit and seg_label must be of the same size
        sampler = OHEMPixelSampler()
        seg_logit = torch.randn(1, 19, 45, 45)
        seg_label = torch.randint(0, 19, size=(1, 1, 89, 89))
        sampler.sample(seg_logit, seg_label)

    sampler = OHEMPixelSampler()
    seg_logit = torch.randn(1, 19, 45, 45)
    seg_label = torch.randint(0, 19, size=(1, 1, 45, 45))
    seg_weight = sampler.sample(seg_logit, seg_label)
    assert seg_weight.shape[0] == seg_logit.shape[0]
    assert seg_weight.shape[1:] == seg_logit.shape[2:]
