import warnings

import torch.nn as nn
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.cnn.bricks.registry import UPSAMPLE_LAYERS as MMCV_UPSAMPLE_LAYERS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)
UPSAMPLE_LAYERS = Registry('upsample layer', parent=MMCV_UPSAMPLE_LAYERS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_upsample_layer(cfg, *args, **kwargs):
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
                `DeconvModule`, `InterpConv`.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in UPSAMPLE_LAYERS:
        raise KeyError(f'Unrecognized upsample type {layer_type}')
    else:
        upsample = UPSAMPLE_LAYERS.get(layer_type)

    if upsample is nn.Upsample:
        cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer
