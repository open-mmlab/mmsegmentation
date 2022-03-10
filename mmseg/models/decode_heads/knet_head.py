# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import (FFN, TRANSFORMER_LAYER,
                                         MultiheadAttention,
                                         build_transformer_layer)

from mmseg.models.builder import HEADS, build_head
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.utils import get_root_logger


@TRANSFORMER_LAYER.register_module()
class KernelUpdator(nn.Module):
    """Dynamic Kernel Updator in Kernel Update Head.

    Args:
        in_channels (int): The number of channels of input feature map.
            Default: 256.
        feat_channels (int): The number of middle-stage channels in
            the kernel updator. Default: 64.
        out_channels (int): The number of output channels.
        gate_sigmoid (bool): Whether use sigmoid function in gate
            mechanism. Default: True.
        gate_norm_act (bool): Whether add normalization and activation
            layer in gate mechanism. Default: False.
        activate_out: Whether add activation after gate mechanism.
            Default: False.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='LN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
    """

    def __init__(
            self,
            in_channels=256,
            feat_channels=64,
            out_channels=None,
            gate_sigmoid=True,
            gate_norm_act=False,
            activate_out=False,
            norm_cfg=dict(type='LN'),
            act_cfg=dict(type='ReLU', inplace=True),
    ):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, update_feature, input_feature):
        """Forward function of KernelUpdator.

        Args:
            update_feature (torch.Tensor): Feature map assembled from
                each group. It would be reshaped with last dimension
                shape: `self.in_channels`.
            input_feature (torch.Tensor): Intermediate feature
                with shape: (N, num_classes, conv_kernel_size**2, channels).
        Returns:
            Tensor: The output tensor of shape (N*C1/C2, K*K, C2), where N is
            the number of classes, C1 and C2 are the feature map channels of
            KernelUpdateHead and KernelUpdator, respectively.
        """

        update_feature = update_feature.reshape(-1, self.in_channels)
        num_proposals = update_feature.size(0)
        # dynamic_layer works for
        # phi_1 and psi_3 in Eq.(4) and (5) of K-Net paper
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, :self.num_params_in].view(
            -1, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels)

        # input_layer works for
        # phi_2 and psi_4 in Eq.(4) and (5) of K-Net paper
        input_feats = self.input_layer(
            input_feature.reshape(num_proposals, -1, self.feat_channels))
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        # `gate_feats` is F^G in K-Net paper
        gate_feats = input_in * param_in.unsqueeze(-2)
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # Gate mechanism. Eq.(5) in original paper.
        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features


@HEADS.register_module()
class KernelUpdateHead(nn.Module):
    """Kernel Update Head in K-Net.

    Args:
        num_classes (int): Number of classes. Default: 150.
        num_ffn_fcs (int): The number of fully-connected layers in
            FFNs. Default: 2.
        num_heads (int): The number of parallel attention heads.
            Default: 8.
        num_mask_fcs (int): The number of fully connected layers for
            mask prediction. Default: 3.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 2048.
        in_channels (int): The number of channels of input feature map.
            Default: 256.
        out_channels (int): The number of output channels.
            Default: 256.
        dropout (float): The Probability of an element to be
            zeroed in MultiheadAttention and FFN. Default 0.0.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        ffn_act_cfg (dict): Config of activation layers in FFN.
            Default: dict(type='ReLU').
        conv_kernel_size (int): The kernel size of convolution in
            Kernel Update Head for dynamic kernel updation.
            Default: 1.
        feat_transform_cfg (dict | None): Config of feature transform.
            Default: None.
        kernel_init (bool): Whether initiate mask kernel in mask head.
            Default: False.
        with_ffn (bool): Whether add FFN in kernel update head.
            Default: True.
        feat_gather_stride (int): Stride of convolution in feature transform.
            Default: 1.
        mask_transform_stride (int): Stride of mask transform.
            Default: 1.
        kernel_updator_cfg (dict): Config of kernel updator.
            Default: dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')).
    """

    def __init__(self,
                 num_classes=150,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_mask_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 conv_kernel_size=1,
                 feat_transform_cfg=None,
                 kernel_init=False,
                 with_ffn=True,
                 feat_gather_stride=1,
                 mask_transform_stride=1,
                 kernel_updator_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN'))):
        super(KernelUpdateHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.dropout = dropout
        self.num_heads = num_heads
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride

        self.attention = MultiheadAttention(in_channels * conv_kernel_size**2,
                                            num_heads, dropout)
        self.attention_norm = build_norm_layer(
            dict(type='LN'), in_channels * conv_kernel_size**2)[1]
        self.kernel_update_conv = build_transformer_layer(kernel_updator_cfg)

        if feat_transform_cfg is not None:
            kernel_size = feat_transform_cfg.pop('kernel_size', 1)
            transform_channels = in_channels
            self.feat_transform = ConvModule(
                transform_channels,
                in_channels,
                kernel_size,
                stride=feat_gather_stride,
                padding=int(feat_gather_stride // 2),
                **feat_transform_cfg)
        else:
            self.feat_transform = None

        if self.with_ffn:
            self.ffn = FFN(
                in_channels,
                feedforward_channels,
                num_ffn_fcs,
                act_cfg=ffn_act_cfg,
                dropout=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        self.fc_mask = nn.Linear(in_channels, out_channels)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.kernel_init:
            logger = get_root_logger()
            logger.info(
                'mask kernel in mask head is normal initialized by std 0.01')
            nn.init.normal_(self.fc_mask.weight, mean=0, std=0.01)

    def forward(self, x, proposal_feat, mask_preds, mask_shape=None):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            x (Tensor): Feature map from FPN with shape
                (batch_size, feature_dimensions, H , W).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)
            mask_preds (Tensor): mask prediction from the former stage in shape
                (batch_size, num_proposals, H, W).

        Returns:
            Tuple: The first tensor is predicted mask with shape
            (N, num_classes, H, W), the second tensor is dynamic kernel
            with shape (N, num_classes, channels, K, K).
        """
        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)

        C, H, W = x.shape[-3:]

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = gather_mask.softmax(dim=1)

        # Group Feature Assembling. Eq.(3) in original paper.
        # einsum is faster than bmm by 30%
        x_feat = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x)

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        proposal_feat = proposal_feat.reshape(N, num_proposals,
                                              self.in_channels,
                                              -1).permute(0, 1, 3, 2)
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)

        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1).permute(1, 0, 2)
        obj_feat = self.attention_norm(self.attention(obj_feat))
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.permute(1, 0, 2)

        # obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1, self.in_channels)

        # FFN
        if self.with_ffn:
            obj_feat = self.ffn_norm(self.ffn(obj_feat))

        mask_feat = obj_feat

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)

        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
            mask_x = x
        # group conv is 5x faster than unfold and uses about 1/5 memory
        # Group conv vs. unfold vs. concat batch, 2.9ms :13.5ms :3.8ms
        # Group conv vs. unfold vs. concat batch, 278 : 1420 : 369
        # but in real training group conv is slower than concat batch
        # so we keep using concat batch.
        # fold_x = F.unfold(
        #     mask_x,
        #     self.conv_kernel_size,
        #     padding=int(self.conv_kernel_size // 2))
        # mask_feat = mask_feat.reshape(N, num_proposals, -1)
        # new_mask_preds = torch.einsum('bnc,bcl->bnl', mask_feat, fold_x)
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape(N, num_proposals, C,
                                      self.conv_kernel_size,
                                      self.conv_kernel_size)
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(
                    mask_x[i:i + 1],
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = torch.cat(new_mask_preds, dim=0)
        new_mask_preds = new_mask_preds.reshape(N, num_proposals, H, W)
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                mask_shape,
                align_corners=False,
                mode='bilinear')

        return new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(
            N, num_proposals, self.in_channels, self.conv_kernel_size,
            self.conv_kernel_size)


@HEADS.register_module()
class IterativeDecodeHead(BaseDecodeHead):
    """K-Net: Towards Unified Image Segmentation.

    This head is the implementation of
    `K-Net:　<https://arxiv.org/abs/2106.14855>`_.

    Args:
        num_stages (int): The number of stages (kernel update heads)
            in IterativeDecodeHead. Default: 3.
        kernel_generate_head:(dict): Config of kernel generate head which
            generate mask predictions, dynamic kernels and class predictions
            for next kernel update heads.
        kernel_update_head (dict): Config of kernel update head which refine
            dynamic kernels and class predictions iteratively.

    """

    def __init__(self, num_stages, kernel_generate_head, kernel_update_head,
                 **kwargs):
        super(BaseDecodeHead, self).__init__(**kwargs)
        assert num_stages == len(kernel_update_head)
        self.num_stages = num_stages
        self.kernel_generate_head = build_head(kernel_generate_head)
        self.kernel_update_head = nn.ModuleList()
        self.align_corners = self.kernel_generate_head.align_corners
        self.num_classes = self.kernel_generate_head.num_classes
        self.input_transform = self.kernel_generate_head.input_transform
        self.ignore_index = self.kernel_generate_head.ignore_index

        for head_cfg in kernel_update_head:
            self.kernel_update_head.append(build_head(head_cfg))

    def forward(self, inputs):
        """Forward function."""
        feats = self.kernel_generate_head._forward_feature(inputs)
        sem_seg = self.kernel_generate_head.cls_seg(feats)
        seg_kernels = self.kernel_generate_head.conv_seg.weight.clone()
        seg_kernels = seg_kernels[None].expand(
            feats.size(0), *seg_kernels.size())

        stage_segs = [sem_seg]
        for i in range(self.num_stages):
            sem_seg, seg_kernels = self.kernel_update_head[i](feats,
                                                              seg_kernels,
                                                              sem_seg)
            stage_segs.append(sem_seg)
        if self.training:
            return stage_segs
        # only return the prediction of the last stage during testing
        return stage_segs[-1]

    def losses(self, seg_logit, seg_label):
        losses = dict()
        for i, logit in enumerate(seg_logit):
            loss = self.kernel_generate_head.losses(logit, seg_label)
            for k, v in loss.items():
                losses[f'{k}.s{i}'] = v

        return losses
