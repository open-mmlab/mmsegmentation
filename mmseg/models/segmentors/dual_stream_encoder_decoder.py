# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class DualStreamEncoderDecoder(BaseSegmentor):
    """Dual-stream encoder-decoder for cross-modality segmentation.

    This segmentor maintains individual backbones for RGB and disparity inputs
    and merges their multi-scale features inside the decode head.

    Args:
        backbone_rgb (ConfigType): Config dict for the RGB backbone.
        backbone_disp (ConfigType): Config dict for the disparity backbone.
        decode_head (ConfigType): Config dict for the decode head that accepts
            two feature lists ``(feats_rgb, feats_disp)``.
        auxiliary_head (OptConfigType): Optional auxiliary head config. The
            auxiliary head receives the concatenated tuple of features similar
            to the decode head. Defaults to None.
        train_cfg (OptConfigType): Training config. Defaults to None.
        test_cfg (OptConfigType): Test config. Defaults to None.
        data_preprocessor (OptConfigType): Data preprocessor config. Defaults
            to None.
        init_cfg (OptMultiConfig): Initialization config. Defaults to None.
    """

    def __init__(
        self,
        backbone_rgb: ConfigType,
        decode_head: ConfigType,
        backbone_disp: Optional[ConfigType] = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone_rgb = MODELS.build(backbone_rgb)
        if backbone_disp is not None:
            self.backbone_disp = MODELS.build(backbone_disp)
            self.use_disp_branch = True
        else:
            self.backbone_disp = None
            self.use_disp_branch = False
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
        if auxiliary_head is None:
            self.auxiliary_head = None
            return
        if isinstance(auxiliary_head, Sequence):
            module_list = []
            for head_cfg in auxiliary_head:
                module_list.append(MODELS.build(head_cfg))
            self.auxiliary_head = nn.ModuleList(module_list)
        else:
            self.auxiliary_head = MODELS.build(auxiliary_head)

    @property
    def with_auxiliary_head(self) -> bool:
        if isinstance(self.auxiliary_head, nn.ModuleList):
            return len(self.auxiliary_head) > 0
        return self.auxiliary_head is not None

    def _forward_auxiliary(self, inputs: Tuple[List[Tensor], List[Tensor]],
                           data_samples: SampleList) -> dict:
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def extract_feat(self, inputs: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Extract multi-scale features from RGB and disparity inputs."""
        rgb = inputs[:, :3, :, :]
        feats_rgb = self.backbone_rgb(rgb)
        if not isinstance(feats_rgb, (list, tuple)):
            feats_rgb = [feats_rgb]
        if self.use_disp_branch and inputs.size(1) > 3:
            disp = inputs[:, 3:4, :, :]
            feats_disp = self.backbone_disp(disp) if self.backbone_disp else []
            if not isinstance(feats_disp, (list, tuple)):
                feats_disp = [feats_disp]
        else:
            feats_disp = [torch.zeros_like(feat) for feat in feats_rgb]
        if not isinstance(feats_rgb, (list, tuple)):
            feats_rgb = [feats_rgb]
        return list(feats_rgb), list(feats_disp)

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        feats = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(
            feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(
            self, inputs: Tuple[List[Tensor], List[Tensor]],
            data_samples: SampleList) -> dict:
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        feats = self.extract_feat(inputs)
        losses = self._decode_head_forward_train(feats, data_samples)
        if self.with_auxiliary_head:
            losses.update(self._forward_auxiliary(feats, data_samples))
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        feats = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(feats, data_samples, self.test_cfg)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None):
        feats = self.extract_feat(inputs)
        return self.decode_head.forward(feats)
