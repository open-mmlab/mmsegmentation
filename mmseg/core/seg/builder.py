from mmseg.utils import build_from_cfg
from .registry import SEG_SAMPLERS
from .sampler import BasSegSampler


def build_seg_sampler(cfg, **default_args):
    if isinstance(cfg, BasSegSampler):
        return cfg
    return build_from_cfg(cfg, SEG_SAMPLERS, default_args)
