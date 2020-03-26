from mmseg.utils import Registry

BACKBONES = Registry('backbone')
HEADS = Registry('head')
LOSSES = Registry('loss')
SEGMENTORS = Registry('segmentor')
