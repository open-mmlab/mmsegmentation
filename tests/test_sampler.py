import pytest
import torch

from mmseg.core import OHEMPixelSampler
from mmseg.models.decode_heads import FCNHead


def _context_for_ohem():
    return FCNHead(in_channels=32, channels=16, num_classes=19)


def test_ohem_sampler():

    with pytest.raises(AssertionError):
        # seg_logit and seg_label must be of the same size
        sampler = OHEMPixelSampler(context=_context_for_ohem())
        seg_logit = torch.randn(1, 19, 45, 45)
        seg_label = torch.randint(0, 19, size=(1, 1, 89, 89))
        sampler.sample(seg_logit, seg_label)

    # test with thresh
    sampler = OHEMPixelSampler(
        context=_context_for_ohem(), thresh=0.7, min_kept=200)
    seg_logit = torch.randn(1, 19, 45, 45)
    seg_label = torch.randint(0, 19, size=(1, 1, 45, 45))
    seg_weight = sampler.sample(seg_logit, seg_label)
    assert seg_weight.shape[0] == seg_logit.shape[0]
    assert seg_weight.shape[1:] == seg_logit.shape[2:]
    assert seg_weight.sum() > 200

    # test w.o thresh
    sampler = OHEMPixelSampler(context=_context_for_ohem(), min_kept=200)
    seg_logit = torch.randn(1, 19, 45, 45)
    seg_label = torch.randint(0, 19, size=(1, 1, 45, 45))
    seg_weight = sampler.sample(seg_logit, seg_label)
    assert seg_weight.shape[0] == seg_logit.shape[0]
    assert seg_weight.shape[1:] == seg_logit.shape[2:]
    assert seg_weight.sum() == 200
