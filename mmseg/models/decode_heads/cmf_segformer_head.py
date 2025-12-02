# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


class DepthwiseSobelConv(nn.Conv2d):
    """Depth-wise convolution initialized with a Sobel/Laplacian kernel."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        base_kernel = torch.tensor([
            [0., 1., 0.],
            [1., -4., 1.],
            [0., 1., 0.],
        ])
        kernel = base_kernel.view(1, 1, 3, 3)
        with torch.no_grad():
            self.weight.copy_(kernel.repeat(self.in_channels, 1, 1, 1))
            if self.bias is not None:
                self.bias.zero_()


class DGFE(nn.Module):
    """Depth-Guided Feature Enhancement."""

    def __init__(
        self,
        rgb_channels: int,
        disp_channels: int,
        reduction: int = 4,
        alpha_init: float = 0.5,
        beta_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.edge_conv = DepthwiseSobelConv(disp_channels)
        self.edge_align = nn.Conv2d(disp_channels, rgb_channels, kernel_size=1)
        mid_channels = max(disp_channels // reduction, 1)
        self.conf_pool = nn.AdaptiveAvgPool2d(1)
        self.conf_mlp = nn.Sequential(
            nn.Linear(disp_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, rgb_channels))
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, rgb: torch.Tensor,
                disp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_feat = self.edge_conv(disp)
        edge_feat = self.edge_align(edge_feat)
        a_edge = torch.tanh(edge_feat)

        pooled = self.conf_pool(disp).flatten(1)
        conf = self.conf_mlp(pooled).view(rgb.size(0), -1, 1, 1)
        a_conf = torch.sigmoid(conf)

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        enhanced = rgb * (1 + alpha * a_edge) * (1 + beta * a_conf)
        return enhanced, a_edge, a_conf


class CMG(nn.Module):
    """Cross-Modality Gate for feature fusion."""

    def __init__(self,
                 rgb_channels: int,
                 disp_channels: int,
                 gamma_init: float = 0.5) -> None:
        super().__init__()
        self.align_disp = nn.Conv2d(disp_channels, rgb_channels, kernel_size=1)
        self.transform = nn.Conv2d(disp_channels, rgb_channels, kernel_size=1)
        self.gate_conv = nn.Conv2d(
            rgb_channels * 2, rgb_channels, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, rgb: torch.Tensor,
                disp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        disp_aligned = self.align_disp(disp)
        gate_input = torch.cat([rgb, disp_aligned], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))
        disp_transformed = self.transform(disp)
        gamma = torch.sigmoid(self.gamma)
        fused = (1 - gamma) * rgb + gamma * (rgb + gate * disp_transformed)
        return fused, gate, disp_transformed


@MODELS.register_module()
class CMFSegFormerHead(BaseDecodeHead):
    """SegFormer-style head with cross-modality fusion."""

    def __init__(
        self,
        in_channels: List[int],
        channels: int,
        disp_in_channels: Optional[List[int]] = None,
        reduction: int = 4,
        alpha_init: float = 0.5,
        beta_init: float = 0.5,
        gamma_init: float = 0.5,
        enable_dgfe: bool = True,
        enable_cmg: bool = True,
        use_disp: bool = True,
        with_boundary: bool = False,
        boundary_loss_weight: float = 0.3,
        geometry_reg_weight: float = 0.0,
        road_class_idx: int = 0,
        **kwargs) -> None:
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_transform='multiple_select',
            **kwargs)
        if disp_in_channels is None:
            disp_in_channels = in_channels
        assert len(disp_in_channels) == len(self.in_channels)
        self.disp_in_channels = disp_in_channels
        self.reduction = reduction
        self.enable_dgfe = enable_dgfe
        self.enable_cmg = enable_cmg
        self.use_disp = use_disp
        self.with_boundary = with_boundary
        self.boundary_loss_weight = boundary_loss_weight
        self.geometry_reg_weight = geometry_reg_weight
        self.road_class_idx = road_class_idx

        self.dgfe_modules = nn.ModuleList()
        self.cmg_modules = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        for rgb_c, disp_c in zip(self.in_channels, self.disp_in_channels):
            self.dgfe_modules.append(
                DGFE(rgb_c, disp_c, reduction, alpha_init, beta_init))
            self.cmg_modules.append(CMG(rgb_c, disp_c, gamma_init))
            self.proj_layers.append(
                ConvModule(
                    in_channels=rgb_c,
                    out_channels=self.channels,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.proj_fuse = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.with_boundary:
            self.boundary_head = nn.Conv2d(self.channels, 1, kernel_size=1)
            self.boundary_loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.latest_results: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
        self.latest_results = {}

    def forward(self, inputs: Tuple[List[torch.Tensor], List[torch.Tensor]]):
        assert isinstance(inputs, (tuple, list)) and len(inputs) == 2, (
            'CMFSegFormerHead expects a tuple of (rgb_feats, disp_feats).')
        feats_rgb, feats_disp = inputs
        feats_rgb = self._transform_inputs(feats_rgb)
        feats_disp = self._transform_inputs(feats_disp)
        assert len(feats_rgb) == len(feats_disp)

        upsampled = []
        edge_list, conf_list, gate_list, disp_trans_list = [], [], [], []
        for idx, (fr, fd) in enumerate(zip(feats_rgb, feats_disp)):
            if self.use_disp and self.enable_dgfe:
                fr_enhanced, a_edge, a_conf = self.dgfe_modules[idx](fr, fd)
            else:
                fr_enhanced = fr
                a_edge = fr.new_zeros(fr.shape)
                a_conf = fr.new_zeros(fr.size(0), fr.size(1), 1, 1)

            if self.use_disp and self.enable_cmg:
                fr_fused, gate, fd_trans = self.cmg_modules[idx](fr_enhanced, fd)
            else:
                fr_fused = fr_enhanced
                gate = fr.new_zeros(fr.shape)
                fd_trans = fr.new_zeros(fr.shape)
            proj = self.proj_layers[idx](fr_fused)
            if proj.shape[2:] != feats_rgb[0].shape[2:]:
                proj = F.interpolate(
                    proj,
                    size=feats_rgb[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            upsampled.append(proj)
            edge_list.append(a_edge)
            conf_list.append(a_conf)
            gate_list.append(gate)
            disp_trans_list.append(fd_trans)

        fused = torch.stack(upsampled, dim=0).sum(dim=0)
        fused = self.proj_fuse(fused)
        feature_for_cls = fused
        if self.dropout is not None:
            feature_for_cls = self.dropout(feature_for_cls)
        seg_logits = self.conv_seg(feature_for_cls)

        boundary_pred = None
        if self.with_boundary:
            boundary_pred = self.boundary_head(feature_for_cls)

        self.latest_results = {
            'fused_feature': fused,
            'a_edge': edge_list,
            'a_conf': conf_list,
            'gate': gate_list,
            'fd_trans': disp_trans_list,
            'boundary_pred': boundary_pred,
        }
        return seg_logits

    def loss_by_feat(self, seg_logits: torch.Tensor,
                     batch_data_samples) -> Dict[str, torch.Tensor]:
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        device = seg_logits.device
        if self.with_boundary and self.latest_results.get('boundary_pred') is not None:
            boundary_pred = self.latest_results['boundary_pred']
            edge_targets = []
            for data_sample in batch_data_samples:
                edge_data = None
                if hasattr(data_sample, 'edge_map'):
                    edge_data = data_sample.edge_map.data
                elif hasattr(data_sample, 'gt_edge_map'):
                    edge_data = data_sample.gt_edge_map.data
                if edge_data is not None:
                    edge = edge_data.float()
                    if edge.dim() == 2:
                        edge = edge.unsqueeze(0).unsqueeze(0)
                    elif edge.dim() == 3 and edge.size(0) != 1:
                        edge = edge.mean(dim=0, keepdim=True).unsqueeze(0)
                    elif edge.dim() == 3:
                        edge = edge.unsqueeze(0)
                    edge_targets.append(edge)
                else:
                    zero_edge = torch.zeros(1, 1, *boundary_pred.shape[2:],
                                             device=device)
                    edge_targets.append(zero_edge)
            edge_target = torch.cat(edge_targets, dim=0).to(device)
            if boundary_pred.shape[2:] != edge_target.shape[2:]:
                boundary_pred = F.interpolate(
                    boundary_pred,
                    size=edge_target.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            losses['loss_edge'] = self.boundary_loss(boundary_pred,
                                                     edge_target) * \
                self.boundary_loss_weight

        if self.geometry_reg_weight > 0:
            road_masks, disp_grads = [], []
            for data_sample in batch_data_samples:
                if hasattr(data_sample, 'road_mask') and hasattr(
                        data_sample, 'disp_grad'):
                    road = data_sample.road_mask.data.float()
                    grad = data_sample.disp_grad.data.float()
                    if road.dim() == 2:
                        road = road.unsqueeze(0).unsqueeze(0)
                    elif road.dim() == 3 and road.size(0) != 1:
                        road = road.mean(dim=0, keepdim=True).unsqueeze(0)
                    elif road.dim() == 3:
                        road = road.unsqueeze(0)
                    if grad.dim() == 2:
                        grad = grad.unsqueeze(0).unsqueeze(0)
                    elif grad.dim() == 3 and grad.size(0) != 1:
                        grad = grad.mean(dim=0, keepdim=True).unsqueeze(0)
                    elif grad.dim() == 3:
                        grad = grad.unsqueeze(0)
                    road_masks.append(road)
                    disp_grads.append(grad)
            if road_masks:
                road_mask = torch.stack(road_masks, dim=0).to(device)
                disp_grad = torch.stack(disp_grads, dim=0).to(device)
                prob = seg_logits.softmax(dim=1)
                road_prob = prob[:, self.road_class_idx:self.road_class_idx + 1]
                weight = torch.exp(-disp_grad.abs())
                geo_loss = (road_prob * weight * road_mask).mean()
                losses['loss_geo'] = geo_loss * self.geometry_reg_weight

        return losses

    def get_latest_results(self) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        return self.latest_results
