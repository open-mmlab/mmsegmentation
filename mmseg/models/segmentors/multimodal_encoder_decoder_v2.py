"""
Multi-Modal EncoderDecoder - MMSeg 1.x Version

Key changes from 0.x:
- Registry: SEGMENTORS -> MODELS
- Base class: 0.x EncoderDecoder -> 1.x EncoderDecoder
- forward_train(img, img_metas, gt) -> loss(inputs, data_samples)
- simple_test/forward_test -> predict(inputs, data_samples)
- img_metas dict -> SegDataSample.metainfo
- gt_semantic_seg -> data_samples[i].gt_sem_seg.data
- train_step removed (handled by mmengine Runner)
- _parse_losses removed (handled by mmengine)
- DataContainer removed
"""
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .encoder_decoder import EncoderDecoder
from ..losses.moe_losses import CombinedMoELoss


@MODELS.register_module()
class MultiModalEncoderDecoderV2(EncoderDecoder):
    """Multi-modal encoder-decoder segmentor with MoE support.

    Supports shared/separate decode heads per modality.

    Args:
        use_moe: Enable MoE in backbone.
        use_modal_bias: Enable modal bias in MoE gating.
        moe_balance_weight: Weight for MoE balance loss.
        moe_diversity_weight: Weight for MoE diversity loss.
        multi_tasks_reweight: Reweighting strategy
            ('equal', 'uncertainty', None).
        dataset_names: List of dataset/modality names.
        decoder_mode: 'shared' or 'separate'.
    """

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 # Custom args
                 use_moe: bool = False,
                 use_modal_bias: bool = True,
                 moe_balance_weight: float = 1.0,
                 moe_diversity_weight: float = 0.01,
                 moe_diversity_similarity: str = 'cosine',
                 multi_tasks_reweight: Optional[str] = None,
                 mtl_sigma_init: float = 1.0,
                 dataset_names: list = None,
                 decoder_mode: str = 'separate'):
        self.dataset_names = dataset_names or ['sar', 'rgb', 'GF']
        self.decoder_mode = decoder_mode
        self._use_moe = use_moe
        self._use_modal_bias = use_modal_bias
        self._multi_tasks_reweight = multi_tasks_reweight
        self._mtl_sigma_init = mtl_sigma_init

        if decoder_mode not in ['shared', 'separate']:
            raise ValueError(
                f"decoder_mode must be 'shared' or 'separate', "
                f"got '{decoder_mode}'")

        # Call grandparent __init__ to skip EncoderDecoder's
        # _init_decode_head / _init_auxiliary_head
        # We need custom initialization for multi-head support
        from .base import BaseSegmentor
        BaseSegmentor.__init__(
            self,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # MoE loss
        if use_moe:
            self.moe_loss_fn = CombinedMoELoss(
                balance_weight=moe_balance_weight,
                diversity_weight=moe_diversity_weight,
                diversity_similarity=moe_diversity_similarity
            )

        # Multi-task reweighting
        self.multi_tasks_reweight = multi_tasks_reweight
        self.mtl_sigma_eps = 1e-6
        if multi_tasks_reweight == 'uncertainty':
            self.mtl_sigma = nn.Parameter(
                torch.full((len(self.dataset_names),),
                           float(mtl_sigma_init)))

    def _init_decode_head(self, decode_head):
        if decode_head is None:
            return

        if self.decoder_mode == 'shared':
            if isinstance(decode_head, dict) and 'type' not in decode_head:
                decode_head = decode_head[self.dataset_names[0]]
            self._shared_decode_head = MODELS.build(decode_head)
            self.align_corners = self._shared_decode_head.align_corners
            self.num_classes = self._shared_decode_head.num_classes
            self.out_channels = self._shared_decode_head.out_channels
        else:
            self.decode_heads = nn.ModuleDict()
            if isinstance(decode_head, dict) and 'type' not in decode_head:
                for name in self.dataset_names:
                    self.decode_heads[name] = MODELS.build(
                        decode_head[name])
            else:
                for name in self.dataset_names:
                    self.decode_heads[name] = MODELS.build(decode_head)

            first_head = self.decode_heads[self.dataset_names[0]]
            self.align_corners = first_head.align_corners
            self.num_classes = first_head.num_classes
            self.out_channels = first_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is None:
            return

        if self.decoder_mode == 'shared':
            if isinstance(auxiliary_head, dict) and 'type' not in auxiliary_head:
                auxiliary_head = auxiliary_head[self.dataset_names[0]]
            self._shared_auxiliary_head = MODELS.build(auxiliary_head)
        else:
            self.auxiliary_heads = nn.ModuleDict()
            if isinstance(auxiliary_head, dict) and 'type' not in auxiliary_head:
                for name in self.dataset_names:
                    self.auxiliary_heads[name] = MODELS.build(
                        auxiliary_head[name])
            else:
                for name in self.dataset_names:
                    self.auxiliary_heads[name] = MODELS.build(
                        auxiliary_head)

    @property
    def decode_head(self):
        if self.decoder_mode == 'shared':
            return getattr(self, '_shared_decode_head', None)
        else:
            if hasattr(self, 'decode_heads') and len(self.decode_heads) > 0:
                return self.decode_heads[self.dataset_names[0]]
            return None

    @property
    def auxiliary_head(self):
        if self.decoder_mode == 'shared':
            return getattr(self, '_shared_auxiliary_head', None)
        else:
            if (hasattr(self, 'auxiliary_heads')
                    and len(self.auxiliary_heads) > 0):
                return self.auxiliary_heads[self.dataset_names[0]]
            return None

    @property
    def with_decode_head(self):
        if self.decoder_mode == 'shared':
            return hasattr(self, '_shared_decode_head')
        else:
            return (hasattr(self, 'decode_heads')
                    and len(self.decode_heads) > 0)

    @property
    def with_auxiliary_head(self):
        if self.decoder_mode == 'shared':
            return hasattr(self, '_shared_auxiliary_head')
        else:
            return (hasattr(self, 'auxiliary_heads')
                    and len(self.auxiliary_heads) > 0)

    def loss(self, inputs, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs: Input images (Tensor or List[Tensor]).
            data_samples: list[SegDataSample] with metainfo and gt_sem_seg.

        Returns:
            dict[str, Tensor]: Loss components.
        """
        # Extract modal types from data_samples
        modal_types = [
            ds.metainfo.get('modal_type', 'rgb') for ds in data_samples
        ]

        # Extract dataset names
        ds_names = [
            ds.metainfo.get('dataset_name',
                            ds.metainfo.get('modal_type', 'rgb'))
            for ds in data_samples
        ]

        # For multi-modal backbone: pass List[Tensor] + modal_types
        if isinstance(inputs, torch.Tensor):
            # Standard tensor - pass directly
            # But backbone expects List[Tensor] for multi-modal
            imgs_list = list(inputs)
        else:
            imgs_list = inputs

        x, moe_balance_loss, expert_features = self.extract_feat(
            imgs_list, modal_types=modal_types)

        losses = dict()

        # MoE losses
        if self._use_moe and moe_balance_loss is not None:
            if expert_features is None:
                losses['loss_moe'] = moe_balance_loss
            else:
                total_moe_loss, moe_loss_dict = self.moe_loss_fn(
                    moe_balance_loss, expert_features)
                losses.update(moe_loss_dict)

        modal_losses = {}

        if self.decoder_mode == 'shared':
            self._loss_shared_mode(
                x, data_samples, ds_names, losses, modal_losses)
        else:
            self._loss_separate_mode(
                x, data_samples, ds_names, losses, modal_losses)

        # Apply multi-task reweighting
        if self.multi_tasks_reweight == 'equal' and modal_losses:
            losses.update(self._apply_equal_reweight(modal_losses))
        elif self.multi_tasks_reweight == 'uncertainty' and modal_losses:
            losses.update(
                self._apply_uncertainty_reweight(modal_losses))

        return losses

    def _loss_shared_mode(self, x, data_samples, ds_names,
                          losses, modal_losses):
        """Compute loss in shared decode head mode."""
        if self.multi_tasks_reweight is not None:
            for dataset_name in sorted(set(ds_names)):
                mask = [n == dataset_name for n in ds_names]
                if not any(mask):
                    continue

                # Filter data_samples for this modality
                modal_ds = [ds for ds, m in zip(data_samples, mask) if m]
                # Filter features
                modal_x = tuple(
                    feat[torch.tensor(mask)] for feat in x)

                loss_decode = self._shared_decode_head.loss(
                    modal_x, modal_ds, self.train_cfg)
                losses.update(add_prefix(
                    loss_decode, f'decode_{dataset_name}'))

                modal_loss = self._sum_loss_dict(loss_decode)

                if self.with_auxiliary_head:
                    loss_aux = self._shared_auxiliary_head.loss(
                        modal_x, modal_ds, self.train_cfg)
                    losses.update(add_prefix(
                        loss_aux, f'aux_{dataset_name}'))
                    modal_loss = modal_loss + self._sum_loss_dict(loss_aux)

                modal_losses[dataset_name] = modal_loss
        else:
            loss_decode = self._shared_decode_head.loss(
                x, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_decode, 'decode'))

            if self.with_auxiliary_head:
                loss_aux = self._shared_auxiliary_head.loss(
                    x, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, 'aux'))

    def _loss_separate_mode(self, x, data_samples, ds_names,
                            losses, modal_losses):
        """Compute loss in separate decode head mode."""
        for dataset_name in sorted(set(ds_names)):
            if dataset_name not in self.decode_heads:
                continue

            mask = [n == dataset_name for n in ds_names]
            if not any(mask):
                continue

            modal_ds = [ds for ds, m in zip(data_samples, mask) if m]
            modal_x = tuple(
                feat[torch.tensor(mask)] for feat in x)

            decode_head = self.decode_heads[dataset_name]
            loss_decode = decode_head.loss(
                modal_x, modal_ds, self.train_cfg)
            losses.update(add_prefix(
                loss_decode, f'decode_{dataset_name}'))

            modal_loss = self._sum_loss_dict(loss_decode)

            if self.with_auxiliary_head:
                aux_head = self.auxiliary_heads[dataset_name]
                loss_aux = aux_head.loss(
                    modal_x, modal_ds, self.train_cfg)
                losses.update(add_prefix(
                    loss_aux, f'aux_{dataset_name}'))
                modal_loss = modal_loss + self._sum_loss_dict(loss_aux)

            modal_losses[dataset_name] = modal_loss

    def extract_feat(self, inputs, modal_types=None):
        """Extract features, supporting multi-modal input.

        Args:
            inputs: List[Tensor] or Tensor
            modal_types: List[str] or None

        Returns:
            x, moe_balance_loss, expert_features
        """
        # Check if backbone supports modal_types
        if modal_types is not None and hasattr(self.backbone, 'forward'):
            import inspect
            sig = inspect.signature(self.backbone.forward)
            if 'modal_types' in sig.parameters:
                backbone_output = self.backbone(
                    inputs, modal_types=modal_types)
            else:
                backbone_output = self.backbone(inputs)
        else:
            backbone_output = self.backbone(inputs)

        moe_balance_loss = None
        expert_features = None

        if isinstance(backbone_output, tuple):
            if len(backbone_output) == 3:
                x, moe_balance_loss, expert_features = backbone_output
            elif len(backbone_output) == 2:
                if (isinstance(backbone_output[1], torch.Tensor)
                        and backbone_output[1].dim() == 0):
                    x, moe_balance_loss = backbone_output
                else:
                    x = backbone_output
            else:
                x = backbone_output
        else:
            x = backbone_output
            if not isinstance(x, tuple):
                x = (x,)

        if self.with_neck:
            x = self.neck(x)

        return x, moe_balance_loss, expert_features

    def encode_decode(self, inputs, batch_img_metas: List[dict]):
        """Encode images and decode into segmentation map."""
        modal_types = [
            meta.get('modal_type', 'rgb') for meta in batch_img_metas
        ]

        if isinstance(inputs, torch.Tensor):
            imgs_list = list(inputs)
        else:
            imgs_list = inputs

        x, _, _ = self.extract_feat(imgs_list, modal_types=modal_types)

        # Select appropriate decode head
        if self.decoder_mode == 'shared':
            decode_head = self._shared_decode_head
        else:
            dataset_name = self._get_dataset_name_from_metas(
                batch_img_metas)
            if dataset_name in self.decode_heads:
                decode_head = self.decode_heads[dataset_name]
            else:
                decode_head = self.decode_head

        seg_logits = decode_head.predict(
            x, batch_img_metas, self.test_cfg)

        return seg_logits

    def predict(self, inputs, data_samples: OptSampleList = None):
        """Predict results from inputs and data samples."""
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs, data_samples: OptSampleList = None):
        """Network forward process (tensor mode)."""
        modal_types = None
        if data_samples is not None:
            modal_types = [
                ds.metainfo.get('modal_type', 'rgb')
                for ds in data_samples
            ]

        if isinstance(inputs, torch.Tensor):
            imgs_list = list(inputs)
        else:
            imgs_list = inputs

        x, _, _ = self.extract_feat(imgs_list, modal_types=modal_types)
        return self.decode_head.forward(x)

    def _get_dataset_name_from_metas(self, batch_img_metas):
        if batch_img_metas and len(batch_img_metas) > 0:
            first_meta = batch_img_metas[0]
            if 'dataset_name' in first_meta:
                return first_meta['dataset_name']
            elif 'modal_type' in first_meta:
                return first_meta['modal_type']
        return self.dataset_names[0]

    @staticmethod
    def _sum_loss_dict(loss_dict):
        total_loss = None
        for loss_name, loss_value in loss_dict.items():
            if 'loss' not in loss_name:
                continue
            if isinstance(loss_value, torch.Tensor):
                current = loss_value
            elif isinstance(loss_value, list):
                current = sum(loss for loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
            total_loss = (current if total_loss is None
                          else total_loss + current)
        if total_loss is None:
            total_loss = torch.tensor(0.0)
        return total_loss

    def _apply_equal_reweight(self, modal_losses):
        if not modal_losses:
            return {}
        n_modals = len(modal_losses)
        total = sum(modal_losses.values()) / n_modals
        reweight_losses = {
            'equal_reweighted_total_loss': total,
        }
        for name in modal_losses:
            reweight_losses[f'equal_weight_{name}'] = torch.tensor(
                1.0 / n_modals, device=total.device)
        return reweight_losses

    def _apply_uncertainty_reweight(self, modal_losses):
        loss_sum = None
        reweight_losses = {}

        for idx, dataset_name in enumerate(self.dataset_names):
            if dataset_name not in modal_losses:
                continue
            loss = modal_losses[dataset_name]
            sigma_sq = self.mtl_sigma[idx] ** 2 + self.mtl_sigma_eps
            weighted = 0.5 / sigma_sq * loss + torch.log1p(sigma_sq)
            loss_sum = (weighted if loss_sum is None
                        else loss_sum + weighted)
            reweight_losses[f'mtl_sigma_{dataset_name}'] = (
                sigma_sq.detach())

        if loss_sum is None:
            return {}

        reweight_losses['reweighted_total_losses'] = loss_sum
        return reweight_losses
