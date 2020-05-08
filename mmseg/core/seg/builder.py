from mmcv.utils import Registry, build_from_cfg

SEG_SAMPLERS = Registry('seg sampler')


def build_seg_sampler(cfg, **default_args):
    return build_from_cfg(cfg, SEG_SAMPLERS, default_args)
