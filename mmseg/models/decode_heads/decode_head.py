from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from mmseg.core import build_seg_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class DecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for DecodeHead

        Args:
            in_channels (int): Input channels.
            channels (int): Channels after modules, before conv_seg.
            drop_out_ratio (float): Ratio of dropout layer. Default: 0.1.
            conv_cfg (dict|None): Config of conv layers. Default: None.
            norm_cfg (dict|None): Config of norm layers. Default: None.
            act_cfg (dict): Config of activation layers.
                Default: dict(type='ReLU')
            num_classes (int): Number of classes. Default: 19.
            classes_weight (Sequence[float]): Different weight of classes.
                Default: None.
            in_index (int|Sequence[int]): Input feature index. Default: -1
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                Default: None.
            loss_decode (dict): Config of decode loss.
                Default: dict(type='CrossEntropyLoss').
            ignore_index (int): The label index to be ignored. Default: 255
            sampler (dict|None): The config of segmentation map sampler.
                Default: None.
            align_corners (bool): align_corners argument of F.interpolate.
                Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 drop_out_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 num_classes=19,
                 classes_weight=None,
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=True):
        super(DecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        if classes_weight is not None:
            assert len(classes_weight) == num_classes
        self.classes_weight = classes_weight
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_seg_sampler(sampler)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if drop_out_ratio > 0:
            self.dropout = nn.Dropout2d(drop_out_ratio)
        else:
            self.dropout = None

    def extra_repr(self):
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
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
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def forward(self, inputs):
        pass

    def get_seg(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def losses(self, seg_logit, seg_label, suffix='decode'):
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        if self.classes_weight is not None:
            classes_weight = seg_logit.new_tensor(self.classes_weight)
        else:
            classes_weight = None
        seg_label = seg_label.squeeze(1)
        loss[f'loss_seg_{suffix}'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            classes_weight=classes_weight,
            ignore_index=self.ignore_index)
        loss[f'acc_seg_{suffix}'] = accuracy(seg_logit, seg_label)
        return loss
