# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS, build_neck


class SceneRelation(BaseModule):
    """Foreground-Scene Relation Module.

    Args:
        scene_relation_in_channels (int): The number of input channels for
            Foreground-Scene Relation Module. Default: 2048.
        scene_relation_channel_list　(List[int]): Number of channels per scale
            in Foreground-Scene Relation Module Default: (256, 256, 256, 256).
        scene_relation_out_channels (int): The number of output channels
            for Foreground-Scene Relation Module. Default: 256.
        scale_aware_proj (bool): Default: True.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 scene_relation_in_channel=2048,
                 scene_relation_channel_list=(256, 256, 256, 256),
                 scene_relation_out_channel=256,
                 scale_aware_proj=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(SceneRelation, self).__init__(init_cfg=init_cfg)
        self.scale_aware_proj = scale_aware_proj
        self.in_channels = scene_relation_in_channel
        self.channel_list = scene_relation_channel_list
        self.out_channel = scene_relation_out_channel

        self.scene_encoder = nn.ModuleList()
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()

        for i in range(len(self.channel_list) if self.scale_aware_proj else 1):
            self.scene_encoder.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channel, 1),
                    build_activation_layer(act_cfg),
                    nn.Conv2d(self.out_channel, self.out_channel, 1)))
        for channel in self.channel_list:
            self.content_encoders.append(
                ConvModule(
                    in_channels=channel,
                    out_channels=self.out_channel,
                    kernel_size=1,
                    bias=True,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.feature_reencoders.append(
                ConvModule(
                    in_channels=channel,
                    out_channels=self.out_channel,
                    kernel_size=1,
                    bias=True,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features):
        content_feats = []
        for i, feature in enumerate(features):
            content_feats.append(self.content_encoders[i](feature))

        scene_feats = []
        relations = []
        if self.scale_aware_proj:
            for layer in self.scene_encoder:
                scene_feats.append(layer(scene_feature))
            for scene_feat, content_feat in zip(scene_feats, content_feats):
                relations.append(
                    self.normalizer(
                        (scene_feat * content_feat).sum(dim=1, keepdim=True)))
        else:
            scene_feat = self.scene_encoder(scene_feature)
            for content_feat in content_feats:
                relations.append(
                    self.normalizer(
                        (scene_feat * content_feat).sum(dim=1, keepdim=True)))

        p_feats = []
        for i, feature in enumerate(features):
            p_feats.append(self.feature_reencoders[i](feature))

        refined_feats = [r * p for r, p in zip(relations, p_feats)]
        return refined_feats


@NECKS.register_module()
class FARNeck(BaseModule):
    """FarSeg Neck.

    This neck is the implementation neck of `Foreground-Aware Relation Network
    for Geospatial Object Segmentation in High Spatial Resolution Remote
    Sensing Imagery <https://arxiv.org/abs/2011.09766>`_. Specifically,
    it includes foreground branch, scene embedding branch and
    Foreground-Scene Relation Module in original paper.

    Args:
        neck_cfg:(dict): Config of neck of foreground branch.
        scene_relation_in_channels (int): The number of input channels for
            Foreground-Scene Relation Module. Default: 2048.
        scene_relation_channel_list　(List[int]): Number of channels per scale
            in Foreground-Scene Relation Module Default: (256, 256, 256, 256).
        scene_relation_out_channels (int): The number of output channels
            for Foreground-Scene Relation Module. Default: 256.
        scale_aware_proj (bool): Default: True.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 neck_cfg=dict(
                     type='FPN',
                     in_channels=[256, 512, 1024, 2048],
                     out_channels=256,
                     num_outs=4),
                 scene_relation_in_channels=2048,
                 scene_relation_channel_list=(256, 256, 256, 256),
                 scene_relation_out_channels=256,
                 scale_aware_proj=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(FARNeck, self).__init__(init_cfg)

        self.foreground_branch = build_neck(neck_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.scene_relation = SceneRelation(
            scene_relation_in_channels,
            scene_relation_channel_list,
            scene_relation_out_channels,
            scale_aware_proj,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == 4, 'Length of input feature \
                                        maps must be 4!'

        scene_feat = self.global_avgpool(inputs[-1])
        fpn_feat_lst = self.foreground_branch(inputs)
        refined_fpn_feat_lst = self.scene_relation(scene_feat, fpn_feat_lst)

        return refined_fpn_feat_lst
