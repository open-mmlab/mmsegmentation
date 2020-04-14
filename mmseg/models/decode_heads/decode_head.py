from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import build_loss
from ..losses import accuracy


class DecodeHead(nn.Module):
    """Base class for DecodeHead"""

    __metaclass__ = ABCMeta

    def __init__(self,
                 in_channels,
                 channels,
                 drop_out_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 num_classes=19,
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255):
        super(DecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if drop_out_ratio > 0:
            self.dropout = nn.Dropout2d(drop_out_ratio)
        else:
            self.dropout = None

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform == 'resize_concat':
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            self.in_channels = sum(in_channels)
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        nn.init.normal_(self.conv_seg.weight, 0, 0.01)
        nn.init.constant_(self.conv_seg.bias, 0)

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def forward(self, inputs):
        pass

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def losses(self,
               seg_logit,
               seg_label,
               seg_weight=None,
               class_weight=None,
               suffix='decode'):
        loss = dict()
        seg_logit = F.interpolate(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=False)
        if seg_weight is not None:
            seg_weight = F.interpolate(
                input=seg_weight,
                size=seg_label.shape[2:],
                mode='nearest',
                align_corners=False)
        if class_weight is not None:
            class_weight = seg_logit.new_tensor(class_weight)
        seg_label = seg_label.squeeze(1).long()
        loss['loss_seg_{}'.format(suffix)] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg_{}'.format(suffix)] = accuracy(seg_logit, seg_label)
        return loss
